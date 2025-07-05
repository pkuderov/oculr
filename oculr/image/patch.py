import numpy as np
from matplotlib import pyplot as plt

from oculr.dataset import to_unified, to_plottable, Dataset


def get_patch_ixs(shape, kernel):
    ksz, kst = kernel
    return np.array([
            [
                [i, j, i+ksz, j+ksz]
                for j in range(0, shape[-1] - ksz + 1, kst)
            ]
            for i in range(0, shape[-2] - ksz + 1, kst)
        ],
        dtype=int
    )


def get_patches(img, patch_ixs, orig_shape):
    img = to_unified(img, orig_shape)
    patches = [
        img[..., up_i:bot_i, up_j:bot_j].ravel()
        for row in patch_ixs
        for up_i, up_j, bot_i, bot_j in row
    ]
    patches = np.stack(patches)
    return patches


def plot_patches(patches, orig_shape):
    n_rows = n_cols = int(np.sqrt(patches.shape[0]))
    fig, axes = plt.subplots(n_rows, n_cols)
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            patch = patches[row*n_cols + col]
            ax.imshow(to_plottable(patch, orig_shape))


def _test_mnist():
    seed = 8041990
    ds = Dataset(seed, 'mnist', grayscale=False, lp_norm=None, debug=True)
    kernel_configs = dict(mnist=(14, 7), cifar=(16, 8))
    ksz, kst = kernel_configs[ds.name]
    print(ksz, kst)
    patch_ixs = get_patch_ixs(ds.image_shape, (ksz, kst))
    print(patch_ixs)

    img = ds.train.images[104]
    plt.imshow(to_plottable(img, ds.image_shape))
    plt.show()

    patches = get_patches(img, patch_ixs, ds.image_shape)
    plot_patches(patches, ds.image_shape)
    plt.show()


if __name__ == '__main__':
    _test_mnist()
