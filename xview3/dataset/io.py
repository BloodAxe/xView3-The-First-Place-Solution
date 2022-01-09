import os
from typing import List, Dict, Tuple, Optional, Union

import cv2
import numpy as np

__all__ = [
    "stack_multichannel_image",
    "read_tiff",
    "read_multichannel_image",
    "read_tiff_with_scaling",
]

from xview3.constants import NODATA_VV_DB

channel_name_to_file_name = {
    "vv": "VV_dB.tif",  # VV-polarization SAR image (dB), 10-meter pixels
    "vh": "VH_dB.tif",  # VH-polarization SAR image (dB), 10-meter pixels
    "bathymetry": "bathymetry.tif",  # bathymetry and elevation image (meters), 500-meter pixels
    "wind_direction": "owiWindDirection.tif",  # wind direction (degrees clockwise from north), 500-meter pixels
    "wind_quality": "owiWindQuality.tif",  # wind quality mask (0 good, 1 medium, 2 low, 3 poor), 500-meter pixels
    "wind_speed": "owiWindSpeed.tif",  # wind speed (meters per second), 500-meter pixels
    "mask": "owiMask.tif",  # land/ice mask (0 valid, 1 land, 2 ice), 500-meter pixels
}


def read_tiff(image_fname: str, crop_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None) -> np.ndarray:
    import rasterio
    from rasterio.windows import Window

    window = None
    if crop_coords is not None:
        (row_start, row_stop), (col_start, col_stop) = crop_coords
        window = Window.from_slices((row_start, row_stop), (col_start, col_stop))

    if not os.path.isfile(image_fname):
        raise FileNotFoundError(image_fname)

    with rasterio.open(image_fname) as f:
        return f.read(1, window=window)


def read_owi_mask_with_scaling(
    image_fname,
    crop_coords: Tuple[Tuple[int, int], Tuple[int, int]],
    s: Union[float, Tuple[float, float]] = 1.0 / 50.0,
) -> np.ndarray:
    """

    :param image:
    :param crop_coords:   (row_start, row_stop), (col_start, col_stop)
    :param s:
    :return:
    """
    if isinstance(s, float):
        sx = sy = s
    else:
        sx, sy = s

    image = read_tiff(image_fname)
    (row_start, row_stop), (col_start, col_stop) = crop_coords

    w = col_stop - col_start
    h = row_stop - row_start
    dsize = w, h

    src = np.array(
        [
            [col_start * sx, row_start * sy],
            [col_stop * sx, row_start * sy],
            [col_start * sx, row_stop * sy],
        ],
        dtype=np.float32,
    )
    # dst = np.array([[0, 0], [w - 1, 0], [0, h - 1]], dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [0, h]], dtype=np.float32)

    M = cv2.getAffineTransform(np.expand_dims(src, 1), np.expand_dims(dst, 1))

    water = image == 0
    land = (image == 1) | (image == 5)
    ice = (image == 2) | (image == 6)

    patch = cv2.warpAffine(
        np.dstack([water, land, ice]).astype(np.float32),
        M,
        dsize=dsize,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return patch


def read_tiff_with_scaling(
    image_fname: str,
    crop_coords: Tuple[Tuple[int, int], Tuple[int, int]],
    s: Union[float, Tuple[float, float]] = 1.0 / 50.0,
) -> np.ndarray:
    """

    :param image_fname:
    :param crop_coords:   (row_start, row_stop), (col_start, col_stop)
    :param s:
    :return:
    """
    if isinstance(s, float):
        sx = sy = s
    else:
        sx, sy = s

    image = read_tiff(image_fname)
    (row_start, row_stop), (col_start, col_stop) = crop_coords

    w = col_stop - col_start
    h = row_stop - row_start
    dsize = w, h

    src = np.array(
        [
            [col_start * sx, row_start * sy],
            [col_stop * sx, row_start * sy],
            [col_start * sx, row_stop * sy],
        ],
        dtype=np.float32,
    )
    # dst = np.array([[0, 0], [w - 1, 0], [0, h - 1]], dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [0, h]], dtype=np.float32)

    M = cv2.getAffineTransform(np.expand_dims(src, 1), np.expand_dims(dst, 1))

    patch = cv2.warpAffine(
        image,
        M,
        dsize=dsize,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=NODATA_VV_DB,
    )
    return patch


def read_multichannel_image(scene_id, channels: List[str], crop_coords=None):
    individual_channels = {}
    for channel in channels:
        if channel in {"diff(vv,vh)", "mean(vv,vh)"}:
            continue

        if channel in {"vv", "vh"}:
            channel_image_fname = os.path.join(scene_id, channel_name_to_file_name[channel])
            img = read_tiff(channel_image_fname, crop_coords=crop_coords).astype(np.float32, copy=False)
        elif channel == "sar":
            img = np.dstack(
                [
                    read_tiff(os.path.join(scene_id, channel_name_to_file_name["vh"]), crop_coords=crop_coords).astype(
                        np.float32, copy=False
                    ),
                    read_tiff(os.path.join(scene_id, channel_name_to_file_name["vv"]), crop_coords=crop_coords).astype(
                        np.float32, copy=False
                    ),
                ]
            )
        elif channel == "mask":
            channel_image_fname = os.path.join(scene_id, channel_name_to_file_name[channel])
            img = read_owi_mask_with_scaling(channel_image_fname, crop_coords=crop_coords)
        else:
            channel_image_fname = os.path.join(scene_id, channel_name_to_file_name[channel])
            img = read_tiff_with_scaling(channel_image_fname, crop_coords=crop_coords)
        img[img == NODATA_VV_DB] = float("nan")
        individual_channels[channel] = img

    return individual_channels


def stack_multichannel_image(individual_channels: Dict[str, np.ndarray], input_channels: List[str]):
    channels = []
    for channel_name in input_channels:
        img = individual_channels[channel_name]
        channels.append(img)

    image = np.dstack(channels)
    return image
