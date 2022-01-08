import numpy as np
import torch
import albumentations as A

__all__ = ["SigmoidNormalization", "channel_name_to_mean_std", "CubicRootNormalization"]

channel_name_to_mean_std = {
    "vv": (0, 50),
    "vh": (0, 50),
    "bathymetry": (
        0,
        1000.0,
    ),
    "wind_speed": (6.81594445, 1.62833557),
}
# VH_dB (array([-26.29277562]), array([4.96274162]))
# VV_dB (array([-16.42577586]), array([4.79392252]))
# bathymetry (array([-1027.55193048]), array([1223.22965922]))
# owiMask (array([0.21731419]), array([0.53000913]))
# owiWindDirection (array([194.57592277]), array([52.55040799]))
# owiWindQuality (array([1.56499957]), array([0.64571367]))
# owiWindSpeed (array([6.81594445]), array([1.62833557]))


class CubicRootNormalization(A.ImageOnlyTransform):
    def __init__(self):
        super().__init__(always_apply=True)

    def apply(self, img, **params) -> np.ndarray:
        return np.cbrt(img)

    def get_transform_init_args(self):
        return tuple()


class SigmoidNormalization(A.ImageOnlyTransform):
    def __init__(self, midpoint=-20, temperature=0.18):
        super().__init__(always_apply=True)
        self.midpoint = midpoint
        self.temperature = temperature

    def apply(self, img, **params):
        x = torch.from_numpy(img)
        xs = (x - self.midpoint) * self.temperature
        return xs.sigmoid().numpy()

    def get_transform_init_args_names(self):
        return ("midpoint", "temperature")
