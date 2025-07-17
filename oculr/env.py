from enum import IntEnum, auto
from functools import partial

import gymnasium
import numpy as np
from gymnasium.vector import VectorEnv

from oculr.dataset import Dataset, to_plottable
from oculr.image.buffer import ImageBuffer, PrefetchedImageBuffer
from oculr.timer import timer
from oculr.util import get_obs_shape, get_pos_range, get_obs, to_one_hot_pos, ensure_2d


class ActionTypes(IntEnum):
    ZOOM = 0
    MOVE = auto()
    ANSWER = auto()


class ImageEnvironment(VectorEnv):
    def __init__(
            self, ds: Dataset, *, num_envs, obs_hw_shape=1, max_time_steps=3,
            seed=None,
            step_reward=0., answer_reward=(1.0, -1.0), zoom_reward=0.0, move_reward=0.0,
            is_eval: bool = False, img_buffer_fn=ImageBuffer,
            termination_policy: str = 'first_guess',
            reset_as_step: bool = False,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.n_classes = ds.n_classes
        self.max_time_steps = max_time_steps
        self.num_envs = self.bsz = num_envs
        # neg: penalty
        self.step_reward = step_reward
        # tuple (correct, incorrect)
        self.answer_reward = answer_reward
        self.zoom_reward = zoom_reward
        self.move_reward = move_reward

        self.img_chw_shape = ds.image_shape
        self.obs_hw_shape = ensure_2d(obs_hw_shape)
        self.obs_chw_shape = get_obs_shape(self.img_chw_shape, self.obs_hw_shape)
        self.pos_range = get_pos_range(self.img_chw_shape, self.obs_hw_shape)

        # total obs vector: <is zoomed out + flatten position 2-hot> + <flatten obs image chw>
        self.obs_size = np.prod(self.obs_chw_shape)
        self.pos_enc_size = 1 + self.pos_range[1].sum()
        self.total_obs_size = self.pos_enc_size + self.obs_size

        self.is_eval = is_eval
        data = ds.test if is_eval else ds.train
        # create buffers
        self.data = img_buffer_fn(
            data.images, data.targets,
            img_chw_shape=self.img_chw_shape, obs_chw_shape=self.obs_chw_shape,
            seed=seed
        )

        # first_guess | until_correct
        self.term_policy = termination_policy
        if self.term_policy == 'first_guess':
            self._step_impl = self.step_first_guess
        elif self.term_policy == 'until_correct':
            self._step_impl = self.step_until_correct
        else:
            raise ValueError(self.term_policy)

        self.use_step_api_for_reset = reset_as_step

        self._timestep = None
        self._pos = None
        self._ixs = np.empty(self.bsz, dtype=int)
        self._img = np.empty((self.bsz, *self.data.images.shape[1:]), dtype=float)
        self._th = np.empty((self.bsz, *self.data.thumbnails.shape[1:]), dtype=float)
        self._tar = np.empty(self.bsz, dtype=int)
        self._done_mask = None

        action_box = [3, self.n_classes, *self.pos_range[1]]
        action_box = np.array(action_box, dtype=int).tolist()

        action_names = [
            'action_type', 'image_classes', 'x_pos', 'y_pos'
        ]
        action_description = list(zip(action_box, action_names))
        action_types_description = dict(
            zoom=int(ActionTypes.ZOOM),
            move=int(ActionTypes.MOVE),
            answer=int(ActionTypes.ANSWER)
        )

        # compliance to gymnasium interface
        # TODO: switch to dict obs and actions
        self.single_observation_space = gymnasium.spaces.Tuple((
            gymnasium.spaces.Box(0.0, 1.0, (self.pos_enc_size,)),
            gymnasium.spaces.Box(0.0, 1.0, (self.obs_size, ))
        ))
        self.single_action_space = gymnasium.spaces.MultiDiscrete(
            action_box, dtype=int, seed=self.rng
        )

        self.observation_space = gymnasium.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = gymnasium.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )
        self.metadata['autoreset_mode'] = gymnasium.vector.AutoresetMode.NEXT_STEP
        self.metadata['action_space_description'] = action_description
        self.metadata['action_box'] = action_box
        self.metadata['action_names'] = action_names
        self.metadata['action_types_description'] = action_types_description

    def reset(self, **kwargs):
        self._timestep = np.zeros(self.bsz, dtype=int)
        self._ixs[:], self._img[:], self._th[:], self._tar[:] = self.data.sample(self.bsz)

        # since on the episode reset an initial obs is thumbnail and position
        # is not exposed (zero-empty two-hot pos), initial values could be any
        self._pos = np.zeros((self.bsz, 2), dtype=int)
        self._done_mask = np.zeros(self.bsz, dtype=bool)

        oh_pos = to_one_hot_pos(self._pos, self.pos_range, is_hidden=self._done_mask)
        obs = self._th.copy()

        info = {}
        if self.use_step_api_for_reset:
            reward = np.zeros(self.bsz)
            terminated = np.zeros_like(reward, dtype=bool)
            info |= dict(
                reward=reward, terminated=terminated,
                truncated=terminated.copy(),
                reset_mask=np.logical_not(terminated),
                n_reset=0,
                n_done=0,
                ep_len_sum=0,
                n_correct=0,
            )

        return (oh_pos, obs), info

    def step(self, action):
        return self._step_impl(action)

    def step_first_guess(self, action):
        # action: (BSZ, 4): what(zoom out|move|answer), n_classes, n_H_pos, n_W_pos
        assert action.shape == (self.bsz, 4)
        zma = action[..., 0]

        # for resetting items the last selected action is ignored
        # (since it's done from the terminal/truncated state)
        reset_mask = self._done_mask
        n_reset = np.count_nonzero(reset_mask)

        self._timestep += 1
        if n_reset > 0:
            # reset what is resetting
            self._timestep[reset_mask] = 0
            (
                self._ixs[reset_mask], self._img[reset_mask],
                self._th[reset_mask], self._tar[reset_mask]
            ) = self.data.sample(n_reset)

        # NB: all three masks have resetting items excluded
        zoom_mask, move_mask, answer_mask = self._split_what_action(zma, reset_mask)

        # resetting do not interfere with both flags
        truncated = self._timestep >= self.max_time_steps
        terminated = answer_mask
        done_mask = np.logical_or(terminated, truncated)
        self._done_mask = done_mask

        n_done = np.count_nonzero(done_mask)
        ep_len_sum = 0 if n_done == 0 else self._timestep[done_mask].sum()

        # 1. Handle answering (=guessing the image class)
        #   fill with default step penalty
        reward = np.full(self.bsz, self.step_reward)
        #   fill resetting items if needed
        if n_reset > 0:
            reward[reset_mask] = 0.0
        #   fill answered with the default "incorrect answer" reward
        reward[answer_mask] = self.answer_reward[1]
        #   fill correct answers
        correct_mask = np.logical_and(answer_mask, action[..., 1] == self._tar)
        n_correct = 0 if n_done == 0 else np.count_nonzero(correct_mask)
        reward[correct_mask] = self.answer_reward[0]
        reward[move_mask] += self.move_reward
        reward[zoom_mask] += self.zoom_reward

        # 2. Handle moving. NB: resetting items inherit old pos from prev episode,
        #    but it's ok since new pos should be selected before being exposed to agent
        self._pos[move_mask] = action[..., 2:][move_mask]

        # what to show: patch or zoomed-out-image (aka thumbnail)
        thumbnail_mask = np.logical_or(zoom_mask, reset_mask)
        # thumbnail_mask = reset_mask
        patch_mask = np.logical_not(thumbnail_mask)

        # encode position (+ is zoomed out mask)
        oh_pos = to_one_hot_pos(self._pos, self.pos_range, thumbnail_mask)

        # fill observation
        obs = np.empty((self.bsz, self.obs_size), float)
        obs[thumbnail_mask] = self._th[thumbnail_mask]
        obs[patch_mask] = self._get_obs(self._img[patch_mask], self._pos[patch_mask])

        # careful: return only copied data (to avoid mutation problems)
        return (
            (oh_pos, obs),
            reward, terminated, truncated,
            dict(
                reset_mask=reset_mask,
                n_reset=n_reset,
                n_done=n_done,
                ep_len_sum=ep_len_sum,
                n_correct=n_correct,
            )
        )

    def step_until_correct(self, action):
        # action: (BSZ, 4): what(zoom out|move|answer), n_classes, n_H_pos, n_W_pos
        assert action.shape == (self.bsz, 4)
        zma = action[..., 0]

        # for resetting items the last selected action is ignored
        # (since it's done from the terminal/truncated state)
        reset_mask = self._done_mask
        n_reset = np.count_nonzero(reset_mask)

        self._timestep += 1
        if n_reset > 0:
            # reset what is resetting
            self._timestep[reset_mask] = 0
            (
                self._ixs[reset_mask], self._img[reset_mask],
                self._th[reset_mask], self._tar[reset_mask]
            ) = self.data.sample(n_reset)

        # NB: all three masks have resetting items excluded
        zoom_mask, move_mask, answer_mask = self._split_what_action(zma, reset_mask)

        # 1. Handle answering (=guessing the image class)
        #   fill with default step penalty
        reward = np.full(self.bsz, self.step_reward)
        #   fill resetting items if needed
        if n_reset > 0:
            reward[reset_mask] = 0.0
        #   fill answered with the default "incorrect answer" reward
        reward[answer_mask] = self.answer_reward[1]
        #   fill correct answers
        correct_mask = np.logical_and(answer_mask, action[..., 1] == self._tar)
        reward[correct_mask] = self.answer_reward[0]

        # resetting do not interfere with both flags
        truncated = self._timestep >= self.max_time_steps
        terminated = correct_mask
        done_mask = np.logical_or(terminated, truncated)
        self._done_mask = done_mask

        n_done = np.count_nonzero(done_mask)
        ep_len_sum = 0 if n_done == 0 else self._timestep[done_mask].sum()
        n_correct = 0 if n_done == 0 else np.count_nonzero(correct_mask)

        # 2. Handle moving. NB: resetting items inherit old pos from prev episode,
        #    but it's ok since new pos should be selected before being exposed to agent
        self._pos[move_mask] = action[..., 2:][move_mask]

        # what to show: patch or zoomed-out-image (aka thumbnail)
        thumbnail_mask = np.logical_or(zoom_mask, reset_mask)
        patch_mask = np.logical_not(thumbnail_mask)

        # encode position (+ is zoomed out mask)
        oh_pos = to_one_hot_pos(self._pos, self.pos_range, thumbnail_mask)

        # fill observation
        obs = np.empty((self.bsz, self.obs_size), float)
        obs[thumbnail_mask] = self._th[thumbnail_mask]
        obs[patch_mask] = self._get_obs(self._img[patch_mask], self._pos[patch_mask])

        # careful: return only copied data (to avoid mutation problems)
        return (
            (oh_pos, obs),
            reward, terminated, truncated,
            dict(
                reset_mask=reset_mask,
                n_reset=n_reset,
                n_done=n_done,
                ep_len_sum=ep_len_sum,
                n_correct=n_correct,
            )
        )

    def _sample_pos(self, n):
        return self.rng.integers(*self.pos_range, (n, 2))

    def _get_obs(self, img, pos):
        return get_obs(img, self.img_chw_shape, self.obs_chw_shape, pos)

    @staticmethod
    def _split_what_action(action, reset_mask):
        a = action.copy()
        a[reset_mask] = -1
        # zma: zoom_mask, move_mask, answer_mask
        zoom = a == ActionTypes.ZOOM
        move = a == ActionTypes.MOVE
        answer = a == ActionTypes.ANSWER
        return zoom, move, answer


def test_env():
    from matplotlib import pyplot as plt

    seed = 8041990
    ds = Dataset('cifar', grayscale=False, lp_norm=None, seed=seed)
    env = ImageEnvironment(ds, num_envs=2, obs_hw_shape=8, seed=None)
    o, info = env.reset()

    plt.imshow(to_plottable(o[1][0], ds.image_shape))
    o, r, terminated, truncated, info = env.step(
        np.array([
            [1, 0, 5, 10],
            [0, 10, 4, 6],
        ])
    )
    print(o, r, terminated, truncated, info)

    o, r, terminated, truncated, info = env.step(
        np.array([
            [1, 0, 5, 10],
            [0, 10, 4, 6],
        ])
    )
    print(info)


def benchmark_env():
    seed = 8041990
    ds = Dataset('cifar', grayscale=False, lp_norm=None, seed=seed)
    env = ImageEnvironment(
        ds, num_envs=64, obs_hw_shape=7, max_time_steps=20, seed=42,
        answer_reward=(1.0, -0.3), step_reward=-0.01,
        img_buffer_fn=partial(PrefetchedImageBuffer, prefetch_size=256),
    )

    state = None
    o, info = env.reset()
    n_steps, print_schedule = 30000, 5000
    ret, ep_len_sum, n_eps, v_loss, pi_loss, acc = 0., 0., 0., 0., 0., 0.
    rng = np.random.default_rng()

    a = np.vstack(
        [
            rng.integers(3, size=env.bsz),
            rng.integers(env.n_classes, size=env.bsz),
            rng.integers(env.pos_range[1][0], size=env.bsz),
            rng.integers(env.pos_range[1][1], size=env.bsz),
        ]
    ).T
    print(a.shape)

    st = timer()
    for i in range(1, n_steps + 1):
        a = np.vstack(
            [
                rng.integers(3, size=env.bsz),
                rng.integers(env.n_classes, size=env.bsz),
                rng.integers(env.pos_range[1][0], size=env.bsz),
                rng.integers(env.pos_range[1][1], size=env.bsz),
            ]
        ).T
        o_next, r, terminated, truncated, info = env.step(a)

        ret += r.mean()
        ep_len_sum += info['ep_len_sum']
        acc += info['n_correct']
        n_eps += info['n_done']

        o = o_next

        if i % print_schedule == 0:
            ret /= print_schedule
            ep_len_sum /= n_eps
            acc /= n_eps / 100.0
            print(f'{i}: {acc:.2f} {ret:.3f}  |  {ep_len_sum:.1f}')
            ret, ep_len_sum, n_eps, v_loss, pi_loss, acc = 0., 0., 0., 0., 0., 0.

    end = timer()
    print(f'{n_steps * env.bsz / (end - st) / 1e+3:.2f} kfps')


if __name__ == '__main__':
    test_env()
    benchmark_env()
