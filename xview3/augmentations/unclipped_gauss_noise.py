import numpy as np
import random
import albumentations as A
import collections

__all__ = ["UnclippedGaussNoise"]


def unclipped_gauss_noise(image, gauss):
    return image.astype(np.float32, copy=False) + gauss.astype(np.float32, copy=False)


class UnclippedGaussNoise(A.ImageOnlyTransform):
    def __init__(self, var_limit=(0.01, 0.1), mean=0, per_channel=True, always_apply=False, p=0.5, image_in_log_space=True):
        super().__init__(always_apply, p)
        if isinstance(var_limit, collections.Iterable) and len(var_limit) == 2:
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = tuple(var_limit)
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")

            self.var_limit = (0, var_limit)
        else:
            raise TypeError("Expected var_limit type to be one of (int, float, tuple, list), got {}".format(type(var_limit)))

        self.mean = A.to_tuple(mean)
        self.per_channel = per_channel
        self.image_in_log_space = image_in_log_space

    def apply(self, img, gauss=None, **params):
        if self.image_in_log_space:
            img = np.power(10, img)
        img = unclipped_gauss_noise(img, gauss=gauss)
        if self.image_in_log_space:
            img = np.log10(img)
        return img

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
        mean = random.uniform(self.mean[0], self.mean[1])

        if self.per_channel:
            gauss = random_state.normal(mean, sigma, image.shape)
        else:
            gauss = random_state.normal(mean, sigma, image.shape[:2])
            if len(image.shape) == 3:
                gauss = np.expand_dims(gauss, -1)

        return {"gauss": gauss.astype(np.float32)}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("var_limit", "per_channel", "mean")
