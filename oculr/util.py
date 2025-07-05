import numpy as np
from numba import jit


def ensure_2d(shape):
    return (shape, shape) if isinstance(shape, int) else shape


def get_pos_range(orig_chw_shape, obs_hw_shape):
    """Get an allowed range of the top left corner positions."""
    c, h, w = orig_chw_shape
    oh, ow = obs_hw_shape
    return np.array([
        [0, 0],
        [max(0, h - oh + 1), max(0, w - ow + 1)]
    ])


def get_obs_shape(orig_chw_shape, obs_hw_shape):
    c, h, w = orig_chw_shape
    oh, ow = obs_hw_shape
    return c, oh, ow


@jit
def get_obs(images, img_chw_shape, obs_chw_shape, poses):
    """Fill flatten obs with img patch (of size obs_window) from img at pos."""
    assert images.shape[0] == poses.shape[0]
    bsz = images.shape[0]
    _, h, w = img_chw_shape
    c, oh, ow = obs_chw_shape
    hw = h * w
    out_obs = np.empty((bsz, c * oh * ow), float)
    for img, pos, obs in zip(images, poses, out_obs):
        j = 0
        st = pos[0] * w + pos[1]
        for _ in range(c):
            t = st
            for _ in range(oh):
                obs[j:j+ow] = img[t:t+ow]
                j += ow
                t += w
            st += hw
    return out_obs


@jit
def to_one_hot_pos(poses, pos_range, is_hidden):
    # NB: concatenated 2 one-hots = two-hot | one-hot if hidden
    max_h, max_w = pos_range[1]
    bsz = poses.shape[0]
    # first bit is for "position is hidden" (if thumbnail is returned)
    res = np.zeros((bsz, 1 + max_h + max_w), float)
    for r, (h, w), hid in zip(res, poses, is_hidden):
        if hid:
            r[0] = 1.0
            continue
        r[1 + h] = 1.0
        r[1 + max_h + w] = 1.0
    return res


@jit
def from_one_hot_pos(poses, pos_range):
    max_h, max_w = pos_range[1]
    res = np.empty((poses.shape[0], 2), np.int32)
    for r, pos in zip(res, poses):
        if pos[0] > 0:
            # (-1, -1) denotes "hidden" position, i.e. showing thumbnail
            r[:] = -1
            continue
        _from_one_hot_pos_single_exact(pos[1:], max_h, r)
    return res


@jit
def _from_one_hot_pos_single_exact(pos, max_h, res):
    i, t = 0, 0
    while t < 2:
        if pos[i] > 0:
            res[t] = i - t * (max_h)
            t += 1
            # it's correct for t = 0 -> 1, but for 1 -> 2 it is not used, so OK
            i = max_h - 1
        i += 1


def test_env_helpers():
    from oculr.dataset import Dataset

    seed = 8041990
    ds = Dataset(seed, 'cifar', grayscale=False, lp_norm=None)

    print(ds.image_shape)
    print(get_pos_range((3, 10, 7), (10, 6)))

    poses = np.array([[0, 0], [2, 1]], dtype=int)
    pos_range = np.array([[0, 0], [3, 4]], dtype=int)
    pos_visible = np.array([0, 0])
    pos_hidden = ~pos_visible

    print(to_one_hot_pos(poses, pos_range, pos_visible))
    print(to_one_hot_pos(poses, pos_range, pos_hidden))

    print(
        from_one_hot_pos(
            to_one_hot_pos(poses, pos_range, pos_visible).copy().astype(int), pos_range
        )
    )
    assert np.array_equal(
        from_one_hot_pos(
            to_one_hot_pos(poses, pos_range, pos_visible).copy().astype(int), pos_range
        ),
        poses
    )
    assert np.all(
        from_one_hot_pos(
            to_one_hot_pos(poses, pos_range, pos_hidden).copy().astype(int), pos_range
        ) == -1
    )


if __name__ == '__main__':
    test_env_helpers()
