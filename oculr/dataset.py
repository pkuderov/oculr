from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class DatasetSplit:
    # raw 2D images
    images: npt.NDArray[float]
    # class indices
    targets: npt.NDArray[int]

    _classes: list[npt.NDArray[int]] | None

    def __init__(self, images, targets, image_shape):
        self.images = images
        self.targets = targets
        self.image_shape = image_shape
        self._classes = None

    def __len__(self):
        return len(self.targets)

    @property
    def n_classes(self):
        return 10

    @property
    def classes(self):
        if self._classes is None:
            self._classes = [np.flatnonzero(self.targets == i) for i in range(self.n_classes)]
        return self._classes

    @property
    def image_size(self):
        return self.images.shape[1]


class Dataset:
    name: str
    grayscale: bool
    train: DatasetSplit
    test: DatasetSplit

    # sds: Sds

    def __init__(
            self, seed: int, ds: str = 'mnist', grayscale: bool = True,
            lp_norm: int = None, debug: bool = False
    ):
        if ds != 'cifar':
            grayscale = True

        self.name = ds
        self.grayscale = grayscale
        image_shape, train, test = _load_dataset(
            seed, ds, grayscale=grayscale, lp_norm=lp_norm, debug=debug
        )

        self.train = DatasetSplit(*train, image_shape)
        self.test = DatasetSplit(*test, image_shape)
        # self.sds = Sds(size=self.image_size, sparsity=1.0)
        self.n_channels = self.image_shape[0]

    @property
    def n_classes(self):
        return 10

    @property
    def image_shape(self):
        return self.train.image_shape

    @property
    def image_size(self):
        return self.train.image_size


def _load_dataset(
        seed: int, ds_name: str, test_size: int | float = 10_000,
        grayscale: bool = True, lp_norm: int = None, debug: bool = False
):
    # normalize the images [0, 255] -> [0, 1]
    normalizer = 255.0

    from pathlib import Path
    cache_path = Path(f'~/data/_cache/{ds_name}{"_gs" if grayscale else ""}.pkl')
    cache_path = cache_path.expanduser()

    if cache_path.exists():
        import pickle
        with cache_path.open('rb') as f:
            ds = pickle.load(f)
            images, targets = ds['images'], ds['targets']
    else:
        from sklearn.datasets import fetch_openml

        supported_datasets = {'mnist': 'mnist_784', 'cifar': 'cifar_10'}
        images, targets = fetch_openml(
            supported_datasets[ds_name], version=1, return_X_y=True, as_frame=False,
            parser='auto'
        )
        images = images.astype(float) / normalizer
        if grayscale and ds_name == 'cifar':
            # convert to grayscale
            print('CONVERTING CIFAR TO GRAYSCALE')
            images = images[:, :1024] * 0.30 + images[:, 1024:2048] * 0.59 + images[:, 2048:] * 0.11

        targets = targets.astype(int)
        import pickle
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open('wb') as f:
            pickle.dump({'images': images, 'targets': targets}, f)

    # normalize
    if lp_norm is not None:
        if lp_norm > 0:
            images /= np.linalg.norm(images, ord=lp_norm, axis=-1, keepdims=True)
        elif lp_norm == 0:
            images -= images.mean(0, keepdims=True)
        elif lp_norm < 0:
            images /= np.linalg.norm(images, ord=-lp_norm, axis=-1).mean()

    shapes = dict(mnist=(28, 28), cifar=(32, 32))
    shape = [3] if not grayscale and ds_name == 'cifar' else [1]
    shape.extend(shapes[ds_name])
    shape = tuple(shape)

    print(f'{ds_name} LOADED images: {images.shape} {shape} | targets: {targets.shape}')

    from sklearn.model_selection import train_test_split
    train_images, test_images, train_targets, test_targets = train_test_split(
        images, targets, random_state=seed, test_size=test_size
    )

    # NB: remove after debug session
    if debug:
        n_trains, n_tests = 15_000, 2_500
        train_images, train_targets = train_images[:n_trains], train_targets[:n_trains]
        test_images, test_targets = test_images[:n_tests], test_targets[:n_tests]

    return shape, (train_images, train_targets), (test_images, test_targets)


def to_unified(img, orig_shape=None, patch_size=None):
    """Makes HWC -> CHW conversion of an image."""
    if img.ndim == 3:
        return img
    assert orig_shape is not None
    psz = patch_size
    c = orig_shape[0]
    shape = [c]
    size = img.shape[0] // c
    if psz is None:
        psz = int(np.sqrt(size))
    if isinstance(psz, int):
        psz = (psz, psz)
    shape.extend(psz)
    return img.reshape(shape)


def to_plottable(img, orig_shape=None, patch_size=None):
    """Makes CHW -> HWC conversion of an image"""
    img = to_unified(img, orig_shape, patch_size)
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    elif img.shape[0] == 1:
        img = img.squeeze()
    return img
