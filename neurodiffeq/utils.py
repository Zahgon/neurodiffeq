import os
import random
import warnings
from pathlib import Path
import numpy as np
import torch
import re


def set_tensor_type(device=None, float_bits=32):
    pass


def safe_mkdir(path):
    pass


def set_seed(seed_value, ignore_numpy=False, ignore_torch=False, ignore_random=False):
    """
    Set the random seed for the numpy, torch, and random packages.

    :param seed_value: The value of seed.
    :type seed_value: int
    :param ignore_numpy: If True, the seed for numpy.random will not be set. Defaults to False.
    :type ignore_numpy: bool
    :param ignore_torch: If True, the seed for torch will not be set. Defaults to False.
    :type ignore_torch: bool
    :param ignore_random: If True, the seed for `random` will not be set. Defaults to False.
    :type ignore_random: bool
    """
    if not ignore_numpy:
        np.random.seed(seed_value)
    if not ignore_torch:
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    if not ignore_random:
        random.seed(seed_value)


def get_residual_info(solution, data, diff_eqs, highest_order=0, detach=True):
    pass


def split_columns(mat):
    pass


def hstack(tensors):
    pass


def vstack(tensors):
    pass
