import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

from oculr.dataset import Dataset, to_plottable


def _ensure_2d(shape):
    return (shape, shape) if isinstance(shape, int) else shape


def _support_flatten(func):
    def wrapper(image, shape, *args, **kwargs):
        image = image.reshape(*image.shape[:-1], *shape)
        result = func(image, *args, **kwargs)
        return result.reshape(*result.shape[:-3], -1)

    return wrapper


@_support_flatten
def resize_image_nn(image, new_shape):
    C, H, W = image.shape[-3:]
    new_H, new_W = _ensure_2d(new_shape)

    # Compute scale factors
    row_idx = (np.arange(new_H) * (H / new_H)).astype(np.int32)
    col_idx = (np.arange(new_W) * (W / new_W)).astype(np.int32)

    # Perform nearest-neighbor sampling
    resized_image = image[..., row_idx[:, None], col_idx]
    return resized_image


@_support_flatten
def resize_image_bilinear(image, new_shape):
    C, H, W = image.shape[-3:]
    new_H, new_W = _ensure_2d(new_shape)

    # Compute scale factors
    row_idx = np.linspace(0, H - 1, new_H)
    col_idx = np.linspace(0, W - 1, new_W)

    row_floor = np.floor(row_idx).astype(int)
    col_floor = np.floor(col_idx).astype(int)
    row_ceil = np.clip(row_floor + 1, 0, H - 1)
    col_ceil = np.clip(col_floor + 1, 0, W - 1)

    row_alpha = row_idx - row_floor
    col_alpha = col_idx - col_floor

    row_alpha = np.expand_dims(row_alpha, 1)
    col_alpha = np.expand_dims(col_alpha, 0)
    row_floor = np.expand_dims(row_floor, 1)
    row_ceil = np.expand_dims(row_ceil, 1)

    top_left = image[..., row_floor, col_floor]
    top_right = image[..., row_floor, col_ceil]
    bottom_left = image[..., row_ceil, col_floor]
    bottom_right = image[..., row_ceil, col_ceil]

    resized_image = (
            (1 - row_alpha) * (1 - col_alpha) * top_left +
            (1 - row_alpha) * col_alpha * top_right +
            row_alpha * (1 - col_alpha) * bottom_left +
            row_alpha * col_alpha * bottom_right
    )
    return resized_image


@_support_flatten
def resize_image_gaussian(imgs, new_shape, sigma_scale: float = 0.5):
    """
    Downscale a CHW NumPy array to (C, new_h, new_w) by:
      1) applying a separable Gaussian blur (sigma ∝ scale)
      2) sampling (nearest) at evenly spaced locations in H and W

    sigma_scale: scalar factor for Gaussian sigma relative to scale;
            typical default = 0.5 (you can increase for heavier blur)
    """
    # 1) unpack shapes & sanity check
    C, H, W = imgs.shape[-3:]
    new_h, new_w = _ensure_2d(new_shape)

    assert new_h <= H and new_w <= W, "Only downsampling (new_h <= H, new_w <= W) is supported."

    # 2) compute scale factors and corresponding sigmas
    scale_h = H / new_h
    scale_w = W / new_w
    sigma_h = sigma_scale * scale_h
    sigma_w = sigma_scale * scale_w

    # 3) determine 1D Gaussian kernel sizes (odd, cover ~±3σ)
    kh = int(2 * np.ceil(3 * sigma_h) + 1)
    kw = int(2 * np.ceil(3 * sigma_w) + 1)

    def gaussian_kernel1d(kernel_size: int, sigma: float) -> np.ndarray:
        ax = np.arange(kernel_size) - (kernel_size - 1) / 2.0
        kernel = np.exp(-0.5 * (ax / sigma) ** 2)
        return kernel / kernel.sum()

    kernel_h = gaussian_kernel1d(kh, sigma_h)  # shape (kh,)
    kernel_w = gaussian_kernel1d(kw, sigma_w)  # shape (kw,)

    pad_h = kh // 2
    pad_w = kw // 2

    # 4) --- VERTICAL BLUR (vectorized over N and C) ---
    # Pad along H axis (axis=2) with 'reflect'; resulting shape = (N, C, H + 2*pad_h, W)
    imgs_padded_v = np.pad(
        imgs,
        ((0, 0), (0, 0), (pad_h, pad_h), (0, 0)),
        mode='reflect'
    )

    # sliding_window_v: shape (N, C, H, W, kh)
    sliding_window_v = sliding_window_view(
        imgs_padded_v, window_shape=kh, axis=2
    )

    # Convolve each length‐kh window with kernel_h via tensordot:
    #   tmp_v has shape (N, C, H, W)
    tmp_v = np.tensordot(sliding_window_v, kernel_h, axes=([4], [0]))

    # 5) --- HORIZONTAL BLUR (vectorized over N and C) ---
    # Pad tmp_v along W axis (axis=3) with 'reflect'; shape = (N, C, H, W + 2*pad_w)
    tmp_padded_h = np.pad(
        tmp_v,
        ((0, 0), (0, 0), (0, 0), (pad_w, pad_w)),
        mode='reflect'
    )

    # sliding_window_h: shape (N, C, H, W, kw)
    sliding_window_h = sliding_window_view(
        tmp_padded_h, window_shape=kw, axis=3
    )

    # Convolve each length‐kw window with kernel_w via tensordot:
    #   imgs_blur has shape (N, C, H, W)
    imgs_blur = np.tensordot(sliding_window_h, kernel_w, axes=([4], [0]))

    # 6) --- VECTORIZED SAMPLING AT NEW GRID (nearest neighbor) ---
    ys = np.linspace(0, H - 1, new_h)  # shape (new_h,)
    xs = np.linspace(0, W - 1, new_w)  # shape (new_w,)
    ys_idx = np.round(ys).astype(int).clip(0, H - 1)  # shape (new_h,)
    xs_idx = np.round(xs).astype(int).clip(0, W - 1)  # shape (new_w,)

    # First pick required rows: partial has shape (N, C, new_h, W)
    partial = imgs_blur[:, :, ys_idx, :]

    # Then pick required columns: out has shape (N, C, new_h, new_w)
    out = partial[:, :, :, xs_idx]

    return out.astype(np.float32)


def test_image_resize():
    seed = 8041990
    ds = Dataset(seed, 'cifar', grayscale=False, lp_norm=None)

    sz = 8
    rng = np.random.default_rng()
    img_range = rng.integers(0, 10_000)
    img_range = img_range, img_range + 9
    plt.imshow(
        to_plottable(
            resize_image_nn(ds.train.images[img_range[0]:img_range[1]], ds.image_shape, sz)[5],
            # resize_image_bilinear(ds.train.images[1:10], ds.image_shape, 32)[5],
            ds.image_shape
        )
    )
    plt.show()

    plt.imshow(
        to_plottable(
            resize_image_bilinear(ds.train.images[img_range[0]:img_range[1]], ds.image_shape, sz)[
                5],
            ds.image_shape
        )
    )
    plt.show()

    plt.imshow(
        to_plottable(
            resize_image_bilinear(ds.train.images[img_range[0]:img_range[1]], ds.image_shape, sz)[
                5],
            ds.image_shape
        )
    )
    plt.show()


if __name__ == '__main__':
    test_image_resize()
