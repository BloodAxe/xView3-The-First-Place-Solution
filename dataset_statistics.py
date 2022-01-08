import os
from collections import defaultdict
from functools import partial

from fire import Fire
from pytorch_toolbelt.datasets import DatasetMeanStdCalculator
from pytorch_toolbelt.utils import fs
from tqdm import tqdm

from xview3.constants import NODATA_VV_DB
from xview3.dataset import read_tiff


def main(data_dir: str):
    dirs = [x for x in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, x))]
    mean_std = defaultdict(partial(DatasetMeanStdCalculator, num_channels=1))
    print(len(dirs))

    for scene_id in tqdm(dirs):
        images = fs.find_in_dir_with_ext(os.path.join(data_dir, scene_id), ".tif")
        for image_fname in images:
            image = read_tiff(image_fname)

            valid_mask = image != NODATA_VV_DB
            slug = fs.id_from_fname(image_fname)
            mean_std[slug].accumulate(image, valid_mask)

    for name, accumulator in mean_std.items():
        print(name, accumulator.compute())


if __name__ == "__main__":
    Fire(main)
