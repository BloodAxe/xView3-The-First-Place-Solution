import albumentations as A
import cv2

from .log_space_wrapper import input_in_log_space

__all__ = ("LogSpaceShiftScaleRotate", "LogSpaceGaussianBlur", "LogSpaceMedianBlur")


class LogSpaceMedianBlur(A.MedianBlur):
    @input_in_log_space
    def apply(self, image, ksize=3, **params):
        return super().apply(
            image,
            ksize=ksize,
            **params,
        )


class LogSpaceGaussianBlur(A.GaussianBlur):
    @input_in_log_space
    def apply(self, image, ksize=3, sigma=0, **params):
        return super().apply(
            image,
            ksize=ksize,
            sigma=sigma,
            **params,
        )


class LogSpaceShiftScaleRotate(A.ShiftScaleRotate):
    @input_in_log_space
    def apply(self, img, angle=0, scale=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        return super().apply(
            img,
            angle=angle,
            scale=scale,
            dx=dx,
            dy=dy,
            interpolation=interpolation,
            **params,
        )
