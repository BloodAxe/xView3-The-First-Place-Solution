from functools import wraps

import numpy as np

__all__ = ["input_in_log_space"]


def input_in_log_space(func):
    """"""

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        img_linear = np.power(10, img)
        result = func(img_linear, *args, **kwargs)
        return np.log10(result)

    return wrapped_function
