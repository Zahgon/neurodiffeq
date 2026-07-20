import torch
from .operators import grad


def _l1_norm(residual, funcs, coords):
    pass


def _l2_norm(residual, funcs, coords):
    pass


def _infinity_norm(residual, funcs, coords):
    pass


def _h1_norm(residual, funcs, coords):
    pass


def _h1_semi_norm(residual, funcs, coords):
    pass


_losses = {
    'l1': _l1_norm,
    'l2': _l2_norm,
    'infinity': _infinity_norm,
    'h1': _h1_norm,
    'h1 semi': _h1_semi_norm,
}
