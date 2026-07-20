import torch
import warnings
import torch.optim as optim
import torch.nn as nn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .networks import FCNN
from .neurodiffeq import safe_diff as diff
from .generators import Generator2D, PredefinedGenerator
from ._version_utils import warn_deprecate_class
from .conditions import IrregularBoundaryCondition
from .conditions import NoCondition, DirichletBVP2D, IBVP1D
from .monitors import Monitor2D
from .solvers import Solution2D
from .solvers import Solver2D
from copy import deepcopy

ExampleGenerator2D = warn_deprecate_class(Generator2D)
PredefinedExampleGenerator2D = warn_deprecate_class(PredefinedGenerator)
Solution = warn_deprecate_class(Solution2D)


def _network_output_2input(net, xs, ys, ith_unit):
    xys = torch.cat((xs, ys), 1)
    nn_output = net(xys)
    if ith_unit is not None:
        return nn_output[:, ith_unit].reshape(-1, 1)
    else:
        return nn_output


def _trial_solution_2input(single_net, nets, xs, ys, conditions):
    pass


def solve2D(
        pde, condition, xy_min=None, xy_max=None,
        net=None, train_generator=None, valid_generator=None, optimizer=None,
        criterion=None, n_batches_train=1, n_batches_valid=4,
        additional_loss_term=None, metrics=None, max_epochs=1000,
        monitor=None, return_internal=False, return_best=False, batch_size=None, shuffle=None,
):
    r"""Train a neural network to solve a PDE with 2 independent variables.

    :param pde:
        The PDE to solve.
        If the PDE is :math:`F(u, x, y) = 0`
        where :math:`u` is the dependent variable and :math:`x` and :math:`y` are the independent variables,
        then `pde` should be a function that maps :math:`(u, x, y)` to :math:`F(u, x, y)`.
    :type pde: callable
    :param condition:
        The initial/boundary condition.
    :type condition: `neurodiffeq.conditions.BaseCondition`
    :param xy_min:
        The lower bound of 2 dimensions.
        If we only care about :math:`x \geq x_0` and :math:`y \geq y_0`,
        then `xy_min` is `(x_0, y_0)`, only needed when ``train_generator`` and ``valid_generator`` are not specified.
        Defaults to None
    :type xy_min: tuple[float, float], optional
    :param xy_max:
        The upper bound of 2 dimensions.
        If we only care about :math:`x \leq x_1` and :math:`y \leq y_1`,
        then `xy_min` is `(x_1, y_1)`, only needed when ``train_generator`` and ``valid_generator`` are not specified.
        Defaults to None
    :type xy_max: tuple[float, float], optional
    :param net:
        The neural network used to approximate the solution.
        Defaults to None.
    :type net: `torch.nn.Module`, optional
    :param train_generator:
        The example generator to generate 1-D training points.
        Default to None.
    :type train_generator: `neurodiffeq.generators.Generator2D`, optional
    :param valid_generator:
        The example generator to generate 1-D validation points.
        Default to None.
    :type valid_generator: `neurodiffeq.generators.Generator2D`, optional
    :param optimizer:
        The optimization method to use for training.
        Defaults to None.
    :type optimizer: `torch.optim.Optimizer`, optional
    :param criterion:
        The loss function to use for training.
        Defaults to None.
    :type criterion: `torch.nn.modules.loss._Loss`, optional
    :param additional_loss_term:
        Extra terms to add to the loss function besides the part specified by `criterion`.
        The input of `additional_loss_term` should be the same as `pde_system`.
    :type additional_loss_term: callable
    :param n_batches_train:
        Number of batches to train in every epoch, where batch-size equals ``train_generator.size``.
        Defaults to 1.
    :type n_batches_train: int, optional
    :param n_batches_valid:
        Number of batches to validate in every epoch, where batch-size equals ``valid_generator.size``.
        Defaults to 4.
    :type n_batches_valid: int, optional
    :param metrics:
        Metrics to keep track of during training.
        The metrics should be passed as a dictionary where the keys are the names of the metrics,
        and the values are the corresponding function.
        The input functions should be the same as `pde` and the output should be a numeric value.
        The metrics are evaluated on both the training set and validation set.
    :type metrics: dict[string, callable]
    :param max_epochs:
        The maximum number of epochs to train.
        Defaults to 1000.
    :type max_epochs: int, optional
    :param monitor:
        The monitor to check the status of neural network during training.
        Defaults to None.
    :type monitor: `neurodiffeq.pde.Monitor2D`, optional
    :param return_internal:
        Whether to return the nets, conditions, training generator, validation generator, optimizer and loss function.
        Defaults to False.
    :type return_internal: bool, optional
    :param return_best:
        Whether to return the nets that achieved the lowest validation loss.
        Defaults to False.
    :type return_best: bool, optional
    :param batch_size:
        **[DEPRECATED and IGNORED]**
        Each batch will use all samples generated.
        Please specify ``n_batches_train`` and ``n_batches_valid`` instead.
    :type batch_size: int
    :param shuffle:
        **[DEPRECATED and IGNORED]**
        Shuffling should be performed by generators.
    :type shuffle: bool
    :return:
        The solution of the PDE. The history of training loss and validation loss.
        Optionally, the nets, conditions, training generator, validation generator, optimizer and loss function.
        The solution is a function that has the signature `solution(xs, ys, as_type)`.
    :rtype: tuple[`neurodiffeq.pde.Solution`, dict] or tuple[`neurodiffeq.pde.Solution`, dict, dict]


    .. note::
        This function is deprecated, use a ``neurodiffeq.solvers.Solver2D`` instead.
    """
    nets = None if not net else [net]
    return solve2D_system(
        pde_system=lambda u, x, y: [pde(u, x, y)], conditions=[condition],
        xy_min=xy_min, xy_max=xy_max, nets=nets,
        train_generator=train_generator, shuffle=shuffle, valid_generator=valid_generator,
        optimizer=optimizer, criterion=criterion, n_batches_train=n_batches_train, n_batches_valid=n_batches_valid,
        additional_loss_term=additional_loss_term, metrics=metrics, batch_size=batch_size,
        max_epochs=max_epochs, monitor=monitor, return_internal=return_internal, return_best=return_best
    )


def solve2D_system(
        pde_system, conditions, xy_min=None, xy_max=None,
        single_net=None, nets=None, train_generator=None, valid_generator=None,
        optimizer=None, criterion=None, n_batches_train=1, n_batches_valid=4,
        additional_loss_term=None, metrics=None, max_epochs=1000,
        monitor=None, return_internal=False, return_best=False, batch_size=None, shuffle=None,
):
    r"""Train a neural network to solve a PDE with 2 independent variables.

    :param pde_system:
        The PDE system to solve.
        If the PDE is :math:`F_i(u_1, u_2, ..., u_n, x, y) = 0`
        where :math:`u_i` is the i-th dependent variable and :math:`x` and :math:`y` are the independent variables,
        then `pde_system` should be a function that maps :math:`(u_1, u_2, ..., u_n, x, y)`
        to a list where the i-th entry is :math:`F_i(u_1, u_2, ..., u_n, x, y)`.
    :type pde_system: callable
    :param conditions:
        The initial/boundary conditions.
        The ith entry of the conditions is the condition that :math:`x_i` should satisfy.
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    :param xy_min:
        The lower bound of 2 dimensions.
        If we only care about :math:`x \geq x_0` and :math:`y \geq y_0`,
        then `xy_min` is `(x_0, y_0)`.
        Only needed when train_generator or valid_generator are not specified.
        Defaults to None
    :type xy_min: tuple[float, float], optional
    :param xy_max:
        The upper bound of 2 dimensions.
        If we only care about :math:`x \leq x_1` and :math:`y \leq y_1`, then `xy_min` is `(x_1, y_1)`.
        Only needed when train_generator or valid_generator are not specified.
        Defaults to None
    :type xy_max: tuple[float, float], optional
    :param single_net:
        The single neural network used to approximate the solution.
        Only one of `single_net` and `nets` should be specified.
        Defaults to None
    :param single_net: `torch.nn.Module`, optional
    :param nets:
        The neural networks used to approximate the solution.
        Defaults to None.
    :type nets: list[`torch.nn.Module`], optional
    :param train_generator:
        The example generator to generate 1-D training points.
        Default to None.
    :type train_generator: `neurodiffeq.generators.Generator2D`, optional
    :param valid_generator:
        The example generator to generate 1-D validation points.
        Default to None.
    :type valid_generator: `neurodiffeq.generators.Generator2D`, optional
    :param optimizer:
        The optimization method to use for training.
        Defaults to None.
    :type optimizer: `torch.optim.Optimizer`, optional
    :param criterion:
        The loss function to use for training.
        Defaults to None.
    :type criterion: `torch.nn.modules.loss._Loss`, optional
    :param n_batches_train:
        Number of batches to train in every epoch, where batch-size equals ``train_generator.size``.
        Defaults to 1.
    :type n_batches_train: int, optional
    :param n_batches_valid:
        Number of batches to validate in every epoch, where batch-size equals ``valid_generator.size``.
        Defaults to 4.
    :type n_batches_valid: int, optional
    :param additional_loss_term:
        Extra terms to add to the loss function besides the part specified by `criterion`.
        The input of `additional_loss_term` should be the same as `pde_system`.
    :type additional_loss_term: callable
    :param metrics:
        Metrics to keep track of during training.
        The metrics should be passed as a dictionary where the keys are the names of the metrics,
        and the values are the corresponding function.
        The input functions should be the same as `pde_system` and the output should be a numeric value.
        The metrics are evaluated on both the training set and validation set.
    :type metrics: dict[string, callable]
    :param max_epochs:
        The maximum number of epochs to train.
        Defaults to 1000.
    :type max_epochs: int, optional
    :param monitor:
        The monitor to check the status of nerual network during training.
        Defaults to None.
    :type monitor: `neurodiffeq.pde.Monitor2D`, optional
    :param return_internal:
        Whether to return the nets, conditions, training generator,
        validation generator, optimizer and loss function.
        Defaults to False.
    :type return_internal: bool, optional
    :param return_best:
        Whether to return the nets that achieved the lowest validation loss.
        Defaults to False.
    :type return_best: bool, optional
    :param batch_size:
        **[DEPRECATED and IGNORED]**
        Each batch will use all samples generated.
        Please specify ``n_batches_train`` and ``n_batches_valid`` instead.
    :type batch_size: int
    :param shuffle:
        **[DEPRECATED and IGNORED]**
        Shuffling should be performed by generators.
    :type shuffle: bool
    :return:
        The solution of the PDE.
        The history of training loss and validation loss.
        Optionally, the nets, conditions, training generator, validation generator, optimizer and loss function.
        The solution is a function that has the signature `solution(xs, ys, as_type)`.
    :rtype: tuple[`neurodiffeq.pde.Solution`, dict] or tuple[`neurodiffeq.pde.Solution`, dict, dict]


    .. note::
        This function is deprecated, use a ``neurodiffeq.solvers.Solver2D`` instead.
    """

    warnings.warn(
        "The `solve2D_system` function is deprecated, use a `neurodiffeq.solvers.Solver2D` instance instead",
        FutureWarning,
    )
    if single_net and nets:
        raise ValueError('Only one of net and nets should be specified')

    if (not single_net) and (not nets):
        single_net = FCNN(
            n_input_units=2,
            n_output_units=len(conditions),
            hidden_units=(32, 32),
            actv=nn.Tanh,
        )

    if single_net:
        for ith, con in enumerate(conditions):
            con.set_impose_on(ith)
        nets = [single_net] * len(conditions)

    if additional_loss_term:
        class CustomSolver2D(Solver2D):
            def additional_loss(self, residual, funcs, coords):
                return additional_loss_term(*funcs, *coords)
    else:
        class CustomSolver2D(Solver2D):
            pass

    solver = CustomSolver2D(
        pde_system=pde_system,
        conditions=conditions,
        xy_min=xy_min,
        xy_max=xy_max,
        nets=nets,
        train_generator=train_generator,
        valid_generator=valid_generator,
        optimizer=optimizer,
        loss_fn=criterion,
        n_batches_train=n_batches_train,
        n_batches_valid=n_batches_valid,
        metrics=metrics,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    solver.fit(max_epochs=max_epochs, monitor=monitor)
    solution = solver.get_solution(copy=True, best=return_best)
    ret = (solution, solver.metrics_history)
    if return_internal:
        params = ['nets', 'conditions', 'train_generator', 'valid_generator', 'optimizer', 'criterion']
        internals = solver.get_internals(params, return_type="dict")
        ret = ret + (internals,)
    return ret


def make_animation(solution, xs, ts):
    r"""Create animation of 1-D time-dependent problems.

    :param solution: Solution function returned by `solve2D` (for a 1-D time-dependent problem).
    :type solution: callable
    :param xs: The locations to evaluate solution.
    :type xs: `numpy.array`
    :param ts: The time points to evaluate solution.
    :type ts: `numpy.array`
    :return: The animation.
    :rtype: `matplotlib.animation.FuncAnimation`
    """

    xx, tt = np.meshgrid(xs, ts)
    sol_net = solution(xx, tt, to_numpy=True)

    def u_gen():
        pass

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)

    umin, umax = sol_net.min(), sol_net.max()
    scale = umax - umin
    ax.set_ylim(umin - scale * 0.1, umax + scale * 0.1)
    ax.set_xlim(xs.min(), xs.max())

    def run(data):
        pass

    return animation.FuncAnimation(
        fig, run, u_gen, blit=True, interval=50, repeat=False
    )



ROUND_TO_ZERO = 1e-7  # in the code below, values lower than ROUND_TO_ZERO are considered zero
K = 5.0
ALPHA = 5.0


class Point:

    def __repr__(self):
        return f'Point({self.loc})'

    def __init__(self, loc):
        self.loc = tuple(float(d) for d in loc)
        self.dim = len(loc)


class DirichletControlPoint(Point):

    def __repr__(self):
        return f'DirichletControlPoint({self.loc}, val={self.val})'

    def __init__(self, loc, val):
        super().__init__(loc)
        self.val = float(val)


class NeumannControlPoint(Point):

    def __repr__(self):
        return f'NeumannControlPoint({self.loc}, val={self.val}, ' + \
               f'normal_vector={self.normal_vector})'

    def __init__(self, loc, val, normal_vector):
        super().__init__(loc)
        self.val = float(val)
        scale = sum(d ** 2 for d in normal_vector) ** 0.5
        self.normal_vector = tuple(d / scale for d in normal_vector)


class CustomBoundaryCondition(IrregularBoundaryCondition):

    def __init__(self, center_point, dirichlet_control_points, neumann_control_points=None):
        super().__init__()

        self.dirichlet_control_points = self._clean_control_points(dirichlet_control_points, center_point)
        self.a_d_interp = InterpolatorCreator.fit_surface(self.dirichlet_control_points)
        self.l_d_interp = InterpolatorCreator.fit_length_factor(self.dirichlet_control_points)

        if neumann_control_points is None:
            neumann_control_points = []
        if len(neumann_control_points) > 0:
            self.neumann_control_points = self._clean_control_points(neumann_control_points, center_point)
            self.g_interp = InterpolatorCreator.fit_surface(self.neumann_control_points)
            self.l_m_interp = InterpolatorCreator.fit_length_factor(self.neumann_control_points)
            self.n_hat_interp = InterpolatorCreator.fit_normal_vector(self.neumann_control_points)
        else:
            self.neumann_control_points = None
            self.g_interp = None
            self.l_m_interp = None
            self.n_hat_interp = None

    def a_d(self, *dimensions):
        return self.a_d_interp.interpolate(dimensions)

    def l_d(self, *dimensions):
        return self.l_d_interp.interpolate(dimensions)

    def g(self, *dimensions):
        return self.g_interp.interpolate(dimensions)

    def l_m(self, *dimensions):
        return self.l_m_interp.interpolate(dimensions)

    def f(self, net, *dimensions):
        return self.l_d(*dimensions) * _network_output_2input(net, *dimensions, self.ith_unit)

    def n_hat(self, *dimensions):
        return self.n_hat_interp.interpolate(dimensions)

    def a_m(self, net, *dimensions):
        if self.neumann_control_points is None:
            return 0.0

        fs = self.f(net, *dimensions)
        a_ds = self.a_d(*dimensions)
        l_ds = self.l_d(*dimensions)
        l_ms = self.l_m(*dimensions)
        n_hats = self.n_hat(*dimensions)

        numer = self.g(*dimensions) - sum(
            nk * (diff(a_ds, d) + diff(fs, d))
            for nk, d in zip(n_hats, dimensions)
        )
        denom = l_ds * sum(
            nk * diff(l_ms, d)
            for nk, d in zip(n_hats, dimensions)
        ) + K * (1 - torch.exp(-ALPHA * l_ms))

        return l_ds * l_ms * numer / denom

    def in_domain(self, *dimensions):
        if self.neumann_control_points is None:
            return self.l_d(*dimensions) > 0.0
        return (self.l_d(*dimensions) > 0.0) & (self.l_m(*dimensions) > 0.0)

    def enforce(self, net, *dimensions):
        return self.a_d(*dimensions) + self.a_m(net, *dimensions) + self.f(net, *dimensions)

    @staticmethod
    def _clean_control_points(control_points, center_point):
        pass


class InterpolatorCreator:

    @staticmethod
    def fit_surface(dirichlet_or_neumann_control_points):
        pass

    @staticmethod
    def fit_length_factor(control_points, radius=0.5):
        pass

    @staticmethod
    def fit_normal_vector(neumann_control_points):
        pass

    @staticmethod
    def _solve_thin_plate_spline(from_points, to_values):
        pass

    @staticmethod
    def _create_circular_targets(control_points, radius):
        pass


class Interpolator:

    def interpolate(self, dimensions):
        raise NotImplementedError

    @staticmethod
    def _interpolate_by_thin_plate_spline(coefs, control_points, dimensions):
        n_pnts = len(control_points)
        to_value_unfinished = torch.zeros_like(dimensions[0])
        for coef, cp in zip(coefs, control_points):
            ri_sq = Interpolator._ri_sq_thin_plate_spline_trainval(cp, dimensions)
            to_value_unfinished += coef * ri_sq * torch.log(ri_sq)
        to_value_unfinished += coefs[n_pnts]
        for j, d in enumerate(dimensions):
            to_value_unfinished += coefs[n_pnts + 1 + j] * d
        to_value = to_value_unfinished
        return to_value

    @staticmethod
    def _ri_sq_thin_plate_spline_pretrain(point_i, point_j, stiffness=0.01):
        pass

    @staticmethod
    def _ri_sq_thin_plate_spline_trainval(point_i, dimensions, stiffness=0.01):
        return sum((d - di) ** 2 for di, d in zip(point_i.loc, dimensions)) + stiffness ** 2


class SurfaceInterpolator(Interpolator):

    def __init__(self, coefs, control_points):
        self.coefs = coefs
        self.control_points = control_points

    def interpolate(self, dimensions):
        return Interpolator._interpolate_by_thin_plate_spline(
            self.coefs, self.control_points, dimensions
        )


class LengthFactorInterpolator(Interpolator):

    def __init__(self, coefs_each_dim, control_points, radius):
        self.coefs_each_dim = coefs_each_dim
        self.control_points = control_points
        self.radius = radius

    def interpolate(self, dimensions):
        dimensions_mapped = tuple(
            Interpolator._interpolate_by_thin_plate_spline(
                coefs_dim, self.control_points, dimensions
            )
            for coefs_dim in self.coefs_each_dim
        )
        return self.radius ** 2 - sum(d ** 2 for d in dimensions_mapped)


class NormalVectorInterpolator(Interpolator):

    def __init__(self, coefs_each_dim, neumann_control_points):
        self.coefs_each_dim = coefs_each_dim
        self.neumann_control_points = neumann_control_points

    def interpolate(self, dimensions):
        dimensions_mapped = tuple(
            Interpolator._interpolate_by_thin_plate_spline(
                coefs_dim, self.neumann_control_points, dimensions
            )
            for coefs_dim in self.coefs_each_dim
        )
        return dimensions_mapped
