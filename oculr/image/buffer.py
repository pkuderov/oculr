import numpy as np

from oculr.dataset import Dataset
from oculr.image.resize import resize_image_nn, _ensure_2d
from oculr.util import _get_obs_shape


class ImageBuffer:
    def __init__(self, images, targets, *, img_chw_shape, obs_chw_shape, seed=None):
        self.rng = np.random.default_rng(seed)
        self.img_chw_shape = img_chw_shape
        self.obs_chw_shape = obs_chw_shape
        self.obs_hw_shape = obs_chw_shape[1:]
        self.images, self.targets = images, targets
        self.thumbnails = resize_image_nn(self.images, self.img_chw_shape, self.obs_hw_shape)
        self._size = len(self.targets)

    def _sample_ixs(self, n):
        return self.rng.integers(0, self._size, n)

    def sample(self, n):
        ixs = self._sample_ixs(n)
        return ixs, self.images[ixs], self.thumbnails[ixs], self.targets[ixs]


class PrefetchedImageBuffer(ImageBuffer):
    def __init__(
            self, images, targets, *, img_chw_shape, obs_chw_shape, prefetch_size, seed=None,
    ):
        super().__init__(
            images, targets, img_chw_shape=img_chw_shape, obs_chw_shape=obs_chw_shape, seed=seed
        )
        self._prefetch_size = prefetch_size
        self._i = prefetch_size
        self._ixs = np.empty(prefetch_size, dtype=int)
        self._img = np.empty((prefetch_size, *self.images.shape[1:]), dtype=self.images.dtype)
        self._th = np.empty(
            (prefetch_size, *self.thumbnails.shape[1:]), dtype=self.images.dtype
        )
        self._tar = np.empty(prefetch_size, dtype=self.images.dtype)

    def sample(self, n):
        self._prefetch(n)
        i = self._i
        ixs = self._ixs[i:i + n]
        img = self._img[i:i + n]
        th = self._th[i:i + n]
        tar = self._tar[i:i + n]
        self._i += n
        return ixs, img, th, tar

    def _prefetch(self, n):
        if self._i + n <= self._prefetch_size:
            return

        # move left to the beginning
        i, left = self._i, self._prefetch_size - self._i

        self._ixs[:left] = self._ixs[i:]
        self._img[:left] = self._img[i:]
        self._th[:left] = self._th[i:]
        self._tar[:left] = self._tar[i:]

        self._ixs[left:] = self._sample_ixs(self._prefetch_size - left)
        ixs = self._ixs[left:]
        self._img[left:] = self.images[ixs]
        self._th[left:] = self.thumbnails[ixs]
        self._tar[left:] = self.targets[ixs]
        self._i = 0


def test_env_buffers():
    seed = 8041990
    ds = Dataset(seed, 'cifar', grayscale=False, lp_norm=None)
    img_buffer = ImageBuffer(
        ds.train.images, ds.train.targets, img_chw_shape=ds.image_shape,
        obs_chw_shape=_get_obs_shape(ds.image_shape, _ensure_2d(12))
    )
    _ = img_buffer.sample(1)
    pref_img_buffer = PrefetchedImageBuffer(
        ds.train.images, ds.train.targets, img_chw_shape=ds.image_shape,
        obs_chw_shape=_get_obs_shape(ds.image_shape, _ensure_2d(12)), prefetch_size=32
    )
    _ = pref_img_buffer.sample(1)


if __name__ == '__main__':
    test_env_buffers()
