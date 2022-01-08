import albumentations as A
import cv2
import numpy as np
from matplotlib import pyplot as plt

__all__ = ["RandomGridShuffle"]


class RandomGridShuffle(A.RandomGridShuffle):
    """
    RandomGridShuffle with keypoints support
    """
    def apply(self, img, tiles=None, **params):
        if tiles is None:
            tiles = []

        return A.swap_tiles_on_image(img, tiles)

    def apply_to_keypoint(self, keypoint, tiles=None, rows=0, cols=0, **params):
        if tiles is None:
            return keypoint

        # for curr_x, curr_y, old_x, old_y, shift_x, shift_y in tiles:
        for (
            current_left_up_corner_row,
            current_left_up_corner_col,
            old_left_up_corner_row,
            old_left_up_corner_col,
            height_tile,
            width_tile,
        ) in tiles:
            x, y = keypoint[:2]

            if (old_left_up_corner_row <= y < (old_left_up_corner_row + height_tile)) and (
                old_left_up_corner_col <= x < (old_left_up_corner_col + width_tile)
            ):
                x = x - old_left_up_corner_col + current_left_up_corner_col
                y = y - old_left_up_corner_row + current_left_up_corner_row
                keypoint = (x, y) + tuple(keypoint[2:])
                break

        return keypoint


def main():
    t = A.Compose(
        [RandomGridShuffle(p=1, grid=(3, 3))],
        keypoint_params=A.KeypointParams(format="xy", label_fields=["labels"]),
    )

    # image = np.zeros((256, 256, 3), dtype=np.uint8)
    image = cv2.imread("lena.png")[..., ::-1].copy()
    # image[::10, :] = 255
    # image[::35, :] = 255
    # image[::51, :] = 255
    # image[:, ::11] = 255
    # image[:, ::24] = 255
    # image[:, ::45] = 255

    kpts = []
    for x in range(20, 512, 20):
        for y in range(20, 512, 20):
            kpts.append((x, y))

    for k in kpts:
        image = cv2.circle(image, k, radius=3, color=(0, 200, 0), thickness=2, lineType=cv2.LINE_AA)

    data = t(image=image, keypoints=kpts, labels=np.zeros(len(kpts)))
    print(len(kpts))
    print(len(data["keypoints"]))
    img = data["image"].copy()

    for k in data["keypoints"]:
        img = cv2.circle(
            img,
            tuple(map(int, k[:2])),
            radius=2,
            color=(200, 0, 200),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    f, ax = plt.subplots(2, 1, figsize=(6, 12))
    ax[0].imshow(image)
    ax[1].imshow(img)
    f.tight_layout()
    f.show()


if __name__ == "__main__":
    main()
