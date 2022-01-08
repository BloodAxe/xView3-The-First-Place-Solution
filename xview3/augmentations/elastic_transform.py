import math
import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from albumentations.augmentations.functional import _maybe_process_in_chunks

__all__ = ["ElasticTransform"]


@A.preserve_shape
def elastic_transform(
    img,
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    value=None,
):
    remap_fn = _maybe_process_in_chunks(
        cv2.remap,
        map1=map_x,
        map2=map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )
    return remap_fn(img)


class ElasticTransform(A.DualTransform):
    def __init__(
        self,
        alpha=1,
        sigma=50,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = A.to_tuple(alpha)
        self.sigma = A.to_tuple(sigma)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def get_transform_init_args_names(self):
        return ("alpha", "sigma", "interpolation", "border_mode", "value", "mask_value")

    def apply(self, img, sigma=0, alpha=0, map_x=None, map_y=None, interpolation=cv2.INTER_LINEAR, **params):
        return elastic_transform(
            img,
            map_x=map_x,
            map_y=map_y,
            interpolation=interpolation,
            border_mode=self.border_mode,
            value=self.value,
        )

    def apply_to_mask(self, img, sigma=0, alpha=0, map_x=None, map_y=None, **params):
        return elastic_transform(
            img,
            map_x=map_x,
            map_y=map_y,
            interpolation=cv2.INTER_NEAREST,
            border_mode=self.border_mode,
            value=self.value,
        )

    def update_params(self, params, **kwargs):
        height, width = kwargs["image"].shape[:2]

        dx = np.zeros((height, width))
        dy = np.zeros((height, width))

        for _ in range(128):
            dx[random.randrange(0, height), random.randrange(0, width)] = random.uniform(self.alpha[0], self.alpha[1])
            dy[random.randrange(0, height), random.randrange(0, width)] = random.uniform(self.alpha[0], self.alpha[1])

        for _ in range(32):
            dx = cv2.blur(dx, (7, 7))
            dy = cv2.blur(dy, (7, 7))

        x, y = np.meshgrid(np.arange(width), np.arange(height))

        params["map_x"] = np.float32(x + dx)
        params["map_y"] = np.float32(y + dy)
        return params

    def apply_to_keypoint(self, keypoint, **params):
        x, y = keypoint[:2]
        map_x, map_y = params["map_x"], params["map_y"]
        mask = np.zeros(map_x.shape[:2], dtype=np.uint8)
        mask[y, x] = 255
        mask = cv2.remap(mask, map_x, map_y, borderMode=cv2.BORDER_CONSTANT, borderValue=0, interpolation=cv2.INTER_LINEAR)
        _, _, _, maxLoc = cv2.minMaxLoc(mask)
        xn, yn = maxLoc
        return (xn, yn) + keypoint[2:]


def main():
    t = A.Compose(
        [ElasticTransform(p=1, alpha=(1000, 5000))],
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
