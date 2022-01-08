from typing import Tuple

import albumentations as A

__all__ = ["CoarseDropout"]


class CoarseDropout(A.CoarseDropout):
    def _keypoint_in_hole(self, keypoint: Tuple, hole: Tuple) -> bool:
        x1, y1, x2, y2 = hole
        x, y = keypoint[:2]
        return x1 <= x < x2 and y1 <= y < y2

    def apply_to_keypoints(self, keypoints, holes=(), **params):
        for hole in holes:
            remaining_keypoints = []
            for kp in keypoints:
                if not self._keypoint_in_hole(kp, hole):
                    remaining_keypoints.append(kp)
            keypoints = remaining_keypoints
        return keypoints


def main():
    from matplotlib import pyplot as plt
    import cv2
    import numpy as np

    t = A.Compose(
        [CoarseDropout(p=1, min_width=32, min_height=32, max_width=128, max_height=128)],
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
    print(kpts)
    print(data["keypoints"])

    for k in data["keypoints"]:
        data["image"] = cv2.circle(
            data["image"],
            tuple(map(int, k[:2])),
            radius=2,
            color=(200, 0, 200),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    f, ax = plt.subplots(2, 1, figsize=(6, 12))
    ax[0].imshow(image)
    ax[1].imshow(data["image"])
    f.tight_layout()
    f.show()


if __name__ == "__main__":
    main()
