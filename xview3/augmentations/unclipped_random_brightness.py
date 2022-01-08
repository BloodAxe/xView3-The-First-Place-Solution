import numpy as np
import random
import albumentations as A

__all__ = ["UnclippedRandomBrightnessContrast"]


def brightness_contrast_adjust_fixed(img, alpha=1.0, beta=0.0):
    if not np.isfinite(img).any():
        return img

    img = img * alpha + beta
    return img


class UnclippedRandomBrightnessContrast(A.ImageOnlyTransform):
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5, image_in_log_space=True, per_channel=False):
        super().__init__(always_apply, p)
        self.brightness_limit = A.to_tuple(brightness_limit)
        self.contrast_limit = A.to_tuple(contrast_limit)
        self.image_in_log_space = image_in_log_space
        self.per_channel = per_channel

    def apply(self, img, alpha=1.0, beta=0.0, **params):
        if self.image_in_log_space:
            img = np.power(10, img)
        img = brightness_contrast_adjust_fixed(img, alpha, beta)
        if self.image_in_log_space:
            img = np.log10(img)
        return img

    def get_params_dependent_on_targets(self, params):
        image = params["image"]

        if self.per_channel:
            num_channels = image.shape[2]
            alphas = [1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]) for _ in range(num_channels)]
            betas = [0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]) for _ in range(num_channels)]

            return {
                "alpha": np.array(alphas, dtype=np.float32),
                "beta": np.array(betas, dtype=np.float32),
            }
        else:
            return {
                "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
                "beta": 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
            }

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "brightness_limit", "contrast_limit", "image_in_log_space", "per_channel"
