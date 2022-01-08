import math

import numpy as np


def pad(vh, rows, cols):
    """
    Pad an image to make it divisible by some block_size.
    Pad on the right and bottom edges so annotations are still usable.
    """
    r, c = vh.shape
    to_rows = math.ceil(r / rows) * rows
    to_cols = math.ceil(c / cols) * cols
    pad_rows = to_rows - r
    pad_cols = to_cols - c
    vh_pad = np.pad(vh, pad_width=((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0)
    return vh_pad, pad_rows, pad_cols


def chip_sar_img(input_img, sz):
    """
    Takes a raster from xView3 as input and outputs
    a set of chips and the coordinate grid for a
    given chip size

    Args:
        input_img (numpy.array): Input image in np.array form
        sz (int): Size of chip (will be sz x sz x # of channlls)

    Returns:
        images: set of image chips
        images_grid: grid coordinates for each chip
    """
    # The input_img is presumed to already be padded
    images = view_as_blocks(input_img, (sz, sz))
    images_grid = images.reshape(int(input_img.shape[0] / sz), int(input_img.shape[1] / sz), sz, sz)
    return images, images_grid


def view_as_blocks(arr, block_size):
    """
    Break up an image into blocks and return array.
    """
    m, n = arr.shape
    M, N = block_size
    return arr.reshape(m // M, M, n // N, N).swapaxes(1, 2).reshape(-1, M, N)
