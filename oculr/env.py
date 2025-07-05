from functools import partial

import numpy as np
from gymnasium.vector import VectorEnv

from oculr.dataset import Dataset, to_plottable
from oculr.image.buffer import ImageBuffer, PrefetchedImageBuffer
from oculr.timer import timer
from oculr.util import get_obs_shape, get_pos_range, get_obs, to_one_hot_pos, ensure_2d


class ImageEnvironment(VectorEnv):
    def __init__(
            self, ds: Dataset, *, num_envs, obs_hw_shape=1, max_time_steps=3, seed=None,
            step_reward=0., answer_reward=(1.0, -1.0),
            is_eval=False, img_buffer_fn=ImageBuffer
    ):
        super().__init__()
        self.n_classes = ds.n_classes
        self.max_time_steps = max_time_steps
        self.num_envs = self.bsz = num_envs
        # neg: penalty
        self.step_reward = step_reward
        # tuple (correct, incorrect)
        self.answer_reward = answer_reward

        self.img_chw_shape = ds.image_shape
        self.obs_hw_shape = ensure_2d(obs_hw_shape)
        self.obs_chw_shape = get_obs_shape(self.img_chw_shape, self.obs_hw_shape)
        self.pos_range = get_pos_range(self.img_chw_shape, self.obs_hw_shape)

        # total obs vector: <is zoomed out + flatten position 2-hot> + <flatten obs image chw>
        self.obs_size = np.prod(self.obs_chw_shape)
        self.total_obs_size = 1 + self.pos_range[1].sum() + self.obs_size

        self.is_eval = is_eval
        data = ds.test if is_eval else ds.train
        # create buffers
        self.data = img_buffer_fn(
            data.images, data.targets,
            img_chw_shape=self.img_chw_shape, obs_chw_shape=self.obs_chw_shape,
            seed=seed
        )

        self._step = None
        self._pos = None
        self._ixs = np.empty(self.bsz, dtype=int)
        self._img = np.empty((self.bsz, *self.data.images.shape[1:]), dtype=float)
        self._th = np.empty((self.bsz, *self.data.thumbnails.shape[1:]), dtype=float)
        self._tar = np.empty(self.bsz, dtype=int)
        self._done_mask = None

    def _sample_pos(self, n):
        return self.rng.integers(*self.pos_range, (n, 2))

    def _get_obs(self, img, pos):
        return get_obs(img, self.img_chw_shape, self.obs_chw_shape, pos)

    def reset(self):
        self._step = np.zeros(self.bsz, dtype=int)
        self._ixs[:], self._img[:], self._th[:], self._tar[:] = self.data.sample(self.bsz)

        # since on the episode reset an initial obs is thumbnail and position
        # is not exposed (zero-empty two-hot pos), initial values could be any
        self._pos = np.zeros((self.bsz, 2), dtype=int)
        self._done_mask = np.zeros(self.bsz, dtype=bool)

        oh_pos = to_one_hot_pos(self._pos, self.pos_range, is_hidden=self._done_mask)
        obs = self._th.copy()

        return (oh_pos, obs), {}

    @staticmethod
    def _split_what_action(action, reset_mask):
        a = action.copy()
        a[reset_mask] = -1
        # zoom_mask, move_mask, guess_mask
        return a == 0, a == 1, a == 2

    def step(self, action):
        # action: (BSZ, 4): what(zoom out|move|answer), n_classes, n_H_pos, n_W_pos
        assert action.shape == (self.bsz, 4)
        zma = action[..., 0]
        # move_action =

        # for resetting items last selected action is ignored (since it's done from the terminal/truncated state)
        reset_mask = self._done_mask
        n_reset = np.count_nonzero(reset_mask)

        self._step += 1
        if n_reset > 0:
            self._step[reset_mask] = 0
            self._ixs[reset_mask], self._img[reset_mask], self._th[reset_mask], self._tar[
                reset_mask] = self.data.sample(n_reset)

        # all three mask have resetting items excluded
        zoom_mask, move_mask, answer_mask = self._split_what_action(zma, reset_mask)

        # resetting do not interfere with both flags
        truncated = self._step >= self.max_time_steps
        terminated = answer_mask
        done_mask = np.logical_or(terminated, truncated)
        self._done_mask = done_mask
        n_done = np.count_nonzero(done_mask)
        ep_len_sum = 0 if n_done == 0 else self._step[done_mask].sum()

        # 1. Handle guessing
        #   fill with default step penalty
        reward = np.full(self.bsz, self.step_reward)
        #   fill asnwered with default incorrect ans reward
        reward[answer_mask] = self.answer_reward[1]
        #   fill correct answers
        correct_mask = np.logical_and(answer_mask, action[..., 1] == self._tar)
        reward[correct_mask] = self.answer_reward[0]
        n_correct = 0 if n_done == 0 else np.count_nonzero(correct_mask)
        #   fill resetting items if needed
        if n_reset > 0:
            reward[reset_mask] = 0.0

        # 2. Handle moving. NB: resetting items inherit old pos from prev episode,
        #    but it's ok since new pos should be chosen before being exposed to agent
        self._pos[move_mask] = action[..., 2:][move_mask]

        thumbnail_mask = np.logical_or(zoom_mask, reset_mask)
        obs_mask = np.logical_not(thumbnail_mask)

        oh_pos = to_one_hot_pos(self._pos, self.pos_range, thumbnail_mask)
        obs = np.empty((self.bsz, self.obs_size), float)
        obs[thumbnail_mask] = self._th[thumbnail_mask]
        obs[obs_mask] = self._get_obs(self._img[obs_mask], self._pos[obs_mask])

        # careful: return only copied data
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


def test_env():
    from matplotlib import pyplot as plt

    seed = 8041990
    ds = Dataset(seed, 'cifar', grayscale=False, lp_norm=None)
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
    ds = Dataset(seed, 'cifar', grayscale=False, lp_norm=None)
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
