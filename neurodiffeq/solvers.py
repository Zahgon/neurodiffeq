import sys
import warnings
import inspect
from inspect import signature
from abc import ABC, abstractmethod
from itertools import chain
from copy import deepcopy
from ordered_set import OrderedSet

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm

from .solvers_utils import PretrainedSolver
from .networks import FCNN
from ._version_utils import deprecated_alias
from .generators import GeneratorSpherical
from .generators import SamplerGenerator
from .generators import Generator1D
from .generators import Generator2D
from .generators import GeneratorND
from .function_basis import RealSphericalHarmonics
from .conditions import BaseCondition
from .neurodiffeq import safe_diff as diff
from .losses import _losses


def _requires_closure(optimizer):
    closure_param = inspect.signature(optimizer.step).parameters.get('closure')
    return closure_param and closure_param.default == inspect._empty


class BaseSolver(ABC, PretrainedSolver):

    @deprecated_alias(criterion='loss_fn')
    def __init__(self, diff_eqs, conditions,
                 nets=None, train_generator=None, valid_generator=None, analytic_solutions=None,
                 optimizer=None, loss_fn=None, n_batches_train=1, n_batches_valid=4,
                 metrics=None, n_input_units=None, n_output_units=None,
                 shuffle=None, batch_size=None):
        if shuffle:
            warnings.warn(
                "param `shuffle` is deprecated and ignored; shuffling should be performed by generators",
                FutureWarning,
            )
        if batch_size is not None:
            warnings.warn(
                "param `batch_size` is deprecated and ignored; specify n_batches_train and n_batches_valid instead",
                FutureWarning,
            )

        self.diff_eqs = diff_eqs
        self.conditions = conditions
        self.n_funcs = len(conditions)
        if nets is None:
            self.nets = [
                FCNN(n_input_units=n_input_units, n_output_units=n_output_units, hidden_units=(32, 32), actv=nn.Tanh)
                for _ in range(self.n_funcs)
            ]
        else:
            self.nets = nets

        if train_generator is None:
            raise ValueError("train_generator must be specified")

        if valid_generator is None:
            raise ValueError("valid_generator must be specified")

        self.metrics_fn = metrics if metrics else {}
        if analytic_solutions:
            warnings.warn(
                'The `analytic_solutions` argument is deprecated and could lead to unstable behavior. '
                'Pass a `metrics` dict instead.',
                FutureWarning,
            )

            def analytic_mse(*args):
                pass

            if 'analytic_mse' in self.metrics_fn:
                warnings.warn(
                    "Ignoring `analytic_solutions` in presence of key 'analytic_mse' in `metrics`",
                    FutureWarning,
                )
            else:
                self.metrics_fn['analytic_mse'] = analytic_mse

        self.metrics_history = {}
        self.metrics_history.update({'train_loss': [], 'valid_loss': []})
        self.metrics_history.update({'train__' + name: [] for name in self.metrics_fn})
        self.metrics_history.update({'valid__' + name: [] for name in self.metrics_fn})

        self.optimizer = optimizer if optimizer else Adam(OrderedSet(chain.from_iterable(n.parameters() for n in self.nets)))
        self._set_loss_fn(loss_fn)

        def make_pair_dict(train=None, valid=None):
            pass

        self.generator = make_pair_dict(
            train=SamplerGenerator(train_generator),
            valid=SamplerGenerator(valid_generator),
        )
        self.n_batches = make_pair_dict(train=n_batches_train, valid=n_batches_valid)
        self._batch = make_pair_dict()
        if self.n_batches['valid'] == 0 and _requires_closure(self.optimizer):
            warnings.warn(f"Setting n_batches_valid=0 will update lowest_loss and best_net with training loss "
                          f"instead of validation loss. "
                          f"This is a problem for {self.optimizer.__class__} optimizer "
                          f"because it updates the parameters before the training loss computed. "
                          f"This leads to potentially worse solution in `best_net`!", RuntimeWarning)
        self.best_nets = None
        self.lowest_loss = None
        self.local_epoch = 0
        self._max_local_epoch = 0
        self._stop_training = False
        self._phase = None

    def _set_loss_fn(self, criterion):
        if criterion is None:
            self.loss_fn = lambda r, f, x: (r ** 2).mean()
        elif isinstance(criterion, nn.modules.loss._Loss):
            self.loss_fn = lambda r, f, x: criterion(r, torch.zeros_like(r))
        elif isinstance(criterion, str):
            self.loss_fn = _losses[criterion.lower()]
        elif callable(criterion):
            self.loss_fn = criterion
        else:
            raise TypeError(f"Unknown type of criterion {type(criterion)}")

    @property
    def global_epoch(self):
        pass

    @property
    def batch(self):
        pass

    @property
    def _batch_examples(self):
        pass

    @property
    def criterion(self):
        warnings.warn(
            f'`{self.__class__.__name__}`.criterion is a deprecated alias for `{self.__class__.__name__}.loss_fn`.'
            f'The alias is only meant to be accessed by certain functions in `neurodiffeq.solver_utils` '
            f'until proper fixes are made; by which time this alias will be removed.'
        )
        return self.loss_fn

    @criterion.setter
    def criterion(self, loss_fn):
        warnings.warn(
            f'`{self.__class__.__name__}`.criterion is a deprecated alias for `{self.__class__.__name__}.loss_fn`.'
            f'The alias is only meant to be accessed by certain functions in `neurodiffeq.solver_utils` '
            f'until proper fixes are made; by which time this alias will be removed.'
        )
        self.loss_fn = loss_fn

    def compute_func_val(self, net, cond, *coordinates):
        r"""Compute the function value evaluated on the points specified by ``coordinates``.

        :param net: The network to be parameterized and evaluated.
        :type net: torch.nn.Module
        :param cond: The condition (a.k.a. parameterization) for the network.
        :type cond: `neurodiffeq.conditions.BaseCondition`
        :param coordinates: A tuple of coordinate components, each with shape = (-1, 1).
        :type coordinates: tuple[torch.Tensor]
        :return: Function values at the sampled points.
        :rtype: torch.Tensor
        """
        return cond.enforce(net, *coordinates)

    def _update_history(self, value, metric_type, key):
        r"""Append a value to corresponding history list.

        :param value: Value to be appended.
        :type value: float
        :param metric_type: Name of the metric. Must be 'loss' or present in ``self.metrics``.
        :type metric_type: str
        :param key: {'train', 'valid'}. Phase of the process.
        :type key: str
        """
        self._phase = key
        if metric_type == 'loss':
            self.metrics_history[f'{key}_{metric_type}'].append(value)
        elif metric_type in self.metrics_fn:
            self.metrics_history[f'{key}__{metric_type}'].append(value)
        else:
            raise KeyError(f"metric '{metric_type}' not specified")

    def _update_train_history(self, value, metric_type):
        pass

    def _update_valid_history(self, value, metric_type):
        pass

    def _generate_batch(self, key):
        r"""Generate the next batch, register in self._batch and return the batch.

        :param key:
            {'train', 'valid'};
            Dict key in ``self._examples``, ``self._batch``, or ``self._batch_start``
        :type key: str
        :return: The generated batch of points.
        :type: List[`torch.Tensor`]
        """
        self._phase = key
        self._batch[key] = [v.reshape(-1, 1) for v in self.generator[key].get_examples()]
        return self._batch[key]

    def _generate_train_batch(self):
        pass

    def _generate_valid_batch(self):
        pass

    def _do_optimizer_step(self, closure=None):
        r"""Optimization procedures after gradients have been computed. Usually ``self.optimizer.step()`` is sufficient.
        At times, users can overwrite this method to perform gradient clipping, etc. Here is an example::

            import itertools
            class MySolver(Solver)
                def _do_optimizer_step(self, closure=None):
                    nn.utils.clip_grad_norm_(itertools.chain([net.parameters() for net in self.nets]), 1.0, 'inf')
                    self.optimizer.step(closure=closure)
        """
        self.optimizer.step(closure=closure)

    def _run_epoch(self, key):
        r"""Run an epoch on train/valid points, update history, and perform an optimization step if key=='train'.

        :param key: {'train', 'valid'}; phase of the epoch
        :type key: str

        .. note::
            The optimization step is only performed after all batches are run.
        """
        if self.n_batches[key] <= 0:
            return
        self._phase = key
        epoch_loss = 0.0
        batch_loss = 0.0
        metric_values = {name: 0.0 for name in self.metrics_fn}

        if key == 'train' and not _requires_closure(self.optimizer):
            self.optimizer.zero_grad()

        for batch_id in range(self.n_batches[key]):
            batch = self._generate_batch(key)

            def closure(zero_grad=True):
                nonlocal batch_loss
                if key == 'train' and zero_grad:
                    self.optimizer.zero_grad()
                funcs = [
                    self.compute_func_val(n, c, *batch) for n, c in zip(self.nets, self.conditions)
                ]

                for name in self.metrics_fn:
                    value = self.metrics_fn[name](*funcs, *batch).item()
                    metric_values[name] += value
                residuals = self.diff_eqs(*funcs, *batch)
                residuals = torch.cat(residuals, dim=1)
                try:
                    loss = self.loss_fn(residuals, funcs, batch) + self.additional_loss(residuals, funcs, batch)
                except TypeError as e:
                    warnings.warn(
                        "You might need to update your code. "
                        "Since v0.4.0; both `criterion` and `additional_loss` requires three inputs: "
                        "`residual`, `funcs`, and `coords`. See documentation for more.", FutureWarning)
                    raise e

                if key == 'train':
                    loss.backward()
                    batch_loss = loss.item()
                return loss

            if key == 'train':
                if _requires_closure(self.optimizer):
                    self._do_optimizer_step(closure=closure)
                else:
                    closure(zero_grad=False)
                epoch_loss += batch_loss
            else:
                epoch_loss += closure().item()

        self._update_history(epoch_loss / self.n_batches[key], 'loss', key)

        if key == 'valid' or self.n_batches['valid'] == 0:
            self._update_best(key)

        if key == 'train' and not _requires_closure(self.optimizer):
            self._do_optimizer_step()

        for name in self.metrics_fn:
            self._update_history(
                metric_values[name] / self.n_batches[key], name, key)

    def run_train_epoch(self):
        r"""Run a training epoch, update history, and perform gradient descent."""
        self._run_epoch('train')

    def run_valid_epoch(self):
        r"""Run a validation epoch and update history."""
        self._run_epoch('valid')

    def _update_best(self, key):
        r"""Update ``self.lowest_loss`` and ``self.best_nets``
        if current training/validation loss is lower than ``self.lowest_loss``
        """
        current_loss = self.metrics_history[key + '_loss'][-1]
        if (self.lowest_loss is None) or current_loss < self.lowest_loss:
            self.lowest_loss = current_loss
            self.best_nets = deepcopy(self.nets)

    def fit(self, max_epochs, callbacks=(), tqdm_file=sys.stderr, **kwargs):
        r"""Run multiple epochs of training and validation, update best loss at the end of each epoch.

        If ``callbacks`` is passed, callbacks are run, one at a time,
        after training, validating and updating best model.

        :param max_epochs: Number of epochs to run.
        :type max_epochs: int
        :param callbacks:
            A list of callback functions.
            Each function should accept the ``solver`` instance itself as its **only** argument.
        :rtype callbacks: list[callable]
        :param tqdm_file:
            File to write tqdm progress bar. If set to None, tqdm is not used at all.
            Defaults to ``sys.stderr``.
        :type tqdm_file: io.StringIO or _io.TextIOWrapper

        .. note::
            1. This method does not return solution, which is done in the ``.get_solution()`` method.
            2. A callback ``cb(solver)`` can set ``solver._stop_training`` to True to perform early stopping.
        """
        self._stop_training = False
        self._max_local_epoch = max_epochs

        monitor = kwargs.pop('monitor', None)
        if monitor:
            warnings.warn("Passing `monitor` is deprecated, "
                          "use a MonitorCallback and pass a list of callbacks instead")
            callbacks = [monitor.to_callback()] + list(callbacks)
        if kwargs:
            raise ValueError(f'Unknown keyword argument(s): {list(kwargs.keys())}')  # pragma: no cover

        if tqdm_file is None:
            loop = range(max_epochs)
        else:
            loop = tqdm(
                range(max_epochs),
                desc='Training Progress',
                colour='blue',
                file=tqdm_file,
                dynamic_ncols=True,
            )

        for local_epoch in loop:
            if self._stop_training:
                break

            self.local_epoch = local_epoch + 1
            self.run_train_epoch()
            self.run_valid_epoch()

            for cb in callbacks:
                cb(self)

    @abstractmethod
    def get_solution(self, copy=True, best=True):
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :return:
            A solution object which can be called.
            To evaluate the solution on certain points,
            you should pass the coordinates vector(s) to the returned solution.
        :rtype: BaseSolution
        """
        pass  # pragma: no cover

    def _get_internal_variables(self):
        r"""Get a dict of all available internal variables.

        :return:
            All available internal variables,
            where keys are variable names and values are the corresponding variables.
        :rtype: dict

        .. note::
            Children classes should inherit all items and optionally include new ones.
        """

        return {
            "metrics": self.metrics_fn,
            "n_batches": self.n_batches,
            "best_nets": self.best_nets,
            "criterion": self.loss_fn,
            "loss_fn": self.loss_fn,
            "conditions": self.conditions,
            "global_epoch": self.global_epoch,
            "lowest_loss": self.lowest_loss,
            "n_funcs": self.n_funcs,
            "nets": self.nets,
            "optimizer": self.optimizer,
            "diff_eqs": self.diff_eqs,
            "generator": self.generator,
            "train_generator": self.generator['train'],
            "valid_generator": self.generator['valid'],
        }

    @deprecated_alias(param_names='var_names')
    def get_internals(self, var_names=None, return_type='list'):
        r"""Return internal variable(s) of the solver

        - If var_names == 'all', return all internal variables as a dict.
        - If var_names is single str, return the corresponding variables.
        - If var_names is a list and return_type == 'list', return corresponding internal variables as a list.
        - If var_names is a list and return_type == 'dict', return a dict with keys in var_names.

        :param var_names: An internal variable name or a list of internal variable names.
        :type var_names: str or list[str]
        :param return_type: {'list', 'dict'}; Ignored if ``var_names`` is a string.
        :type return_type: str
        :return: A single variable, or a list/dict of internal variables as indicated above.
        :rtype: list or dict or any
        """

        available_variables = self._get_internal_variables()

        if var_names == "all" or var_names is None:
            return available_variables

        if isinstance(var_names, str):
            return available_variables[var_names]

        if return_type == 'list':
            return [available_variables[name] for name in var_names]
        elif return_type == "dict":
            return {name: available_variables[name] for name in var_names}
        else:
            raise ValueError(f"unrecognized return_type = {return_type}")

    def additional_loss(self, residual, funcs, coords):
        r"""Additional loss terms for training. This method is to be overridden by subclasses.
        This method can use any of the internal variables: self.nets, self.conditions, self.global_epoch, etc.

        :param residual: Residual tensor of differential equation. It has shape (N_SAMPLES, N_EQUATIONS)
        :type residual: torch.Tensor
        :param funcs:
            Outputs of the networks after parameterization.
            There are ``len(nets)`` entries in total. Each entry is a tensor of shape (N_SAMPLES, N_OUTPUT_UNITS).
        :type funcs: List[torch.Tensor]
        :param coords:
            Inputs to the networks; a.k.a. the spatio-temporal coordinates of the system.
            There are ``N_COORDS`` entries in total. Each entry is a tensor of shape (N_SAMPLES, 1).
        :type coords: List[torch.Tensor]
        :return: Additional loss. Must be a ``torch.Tensor`` of empty shape (scalar).
        :rtype: torch.Tensor
        """
        return 0.0

    def get_residuals(self, *coords, to_numpy=False, best=True, no_reshape=False):
        pass


class BaseSolution(ABC):

    def __init__(self, nets, conditions):
        if nets is None:
            raise RuntimeError("The nets cannot be None, check if you disabled validation "
                               "and used `best`=True with `get_solution` / `get_residual`")
        if isinstance(nets, nn.Module):
            self.nets = [nets] * len(conditions)
        else:
            self.nets = nets
        self.conditions = conditions

    @abstractmethod
    def _compute_u(self, net, condition, *coords):
        pass  # pragma: no cover

    @deprecated_alias(as_type='to_numpy')
    def __call__(self, *coords, to_numpy=False, no_reshape=False):
        r"""Evaluate the solution at certain points.

        :param coords: tuple of coordinate tensors, each of shape (n_samples, 1)
        :type coords: Tuple[`torch.Tensor`]
        :param to_numpy:
            If set to True, the call returns a ``numpy.ndarray`` instead of ``torch.Tensor``.
            Defaults to False.
        :type to_numpy: bool
        :param no_reshape: If set to True, no reshaping will be performed on output. Defaults to False.
        :type no_reshape: bool
        :return:
            Dependent variables evaluated at given points.
            The shape of output will be that of the first input coordinate, unless `no_reshape` is set to True.
        :rtype: list[`torch.Tensor` or `numpy.array`] or `torch.Tensor` or `numpy.array`
        """
        coords = [c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in coords]
        original_shape = coords[0].shape
        coords = [c.reshape(-1, 1) for c in coords]
        if isinstance(to_numpy, str):
            if to_numpy == 'tf' or to_numpy == 'torch':
                to_numpy = False
            elif to_numpy == 'np':
                to_numpy = True
            else:
                raise ValueError(f"Unrecognized `as_type` option: '{to_numpy}'")

        us = [
            self._compute_u(net, con, *coords)
            for con, net in zip(self.conditions, self.nets)
        ]
        if not no_reshape:
            us = [u.reshape(*original_shape) for u in us]
        if to_numpy:
            us = [u.detach().cpu().numpy() for u in us]

        return us if len(self.nets) > 1 else us[0]


class GenericSolution(BaseSolution):
    def _compute_u(self, net, condition, *coords):
        return condition.enforce(net, *coords)


class GenericSolver(BaseSolver):
    def get_solution(self, copy=True, best=True):
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :return:
            A solution object which can be called.
            To evaluate the solution on certain points,
            you should pass the coordinates vector(s) to the returned solution.
        :rtype: BaseSolution
        """
        nets = self.best_nets if best else self.nets
        conditions = self.conditions
        if copy:
            nets = deepcopy(nets)
            conditions = deepcopy(conditions)

        return GenericSolution(nets, conditions)


class SolverSpherical(BaseSolver):

    def __init__(self, pde_system, conditions, r_min=None, r_max=None,
                 nets=None, train_generator=None, valid_generator=None, analytic_solutions=None,
                 optimizer=None, loss_fn=None, n_batches_train=1, n_batches_valid=4, metrics=None, enforcer=None,
                 n_output_units=1,
                 shuffle=None, batch_size=None):

        if train_generator is None or valid_generator is None:
            if r_min is None or r_max is None:
                raise ValueError(f"Either generator is not provided, r_min and r_max should be both provided: "
                                 f"got r_min={r_min}, r_max={r_max}, train_generator={train_generator}, "
                                 f"valid_generator={valid_generator}")

        if train_generator is None:
            train_generator = GeneratorSpherical(512, r_min, r_max, method='equally-spaced-noisy')

        if valid_generator is None:
            valid_generator = GeneratorSpherical(512, r_min, r_max, method='equally-spaced-noisy')

        self.r_min, self.r_max = r_min, r_max
        self.enforcer = enforcer

        super(SolverSpherical, self).__init__(
            diff_eqs=pde_system,
            conditions=conditions,
            nets=nets,
            train_generator=train_generator,
            valid_generator=valid_generator,
            analytic_solutions=analytic_solutions,
            optimizer=optimizer,
            loss_fn=loss_fn,
            n_batches_train=n_batches_train,
            n_batches_valid=n_batches_valid,
            metrics=metrics,
            n_input_units=3,
            n_output_units=n_output_units,
            shuffle=shuffle,
            batch_size=batch_size,
        )

    def _auto_enforce(self, net, cond, *coordinates):
        r"""Enforce condition on network with inputs. If self.enforcer is set, use it.
        Otherwise, fill cond.enforce() with as many arguments as needed.

        :param net: Network for parameterized solution.
        :type net: torch.nn.Module
        :param cond: Condition (a.k.a. parameterization) for the network.
        :type cond: `neurodiffeq.conditions.BaseCondition`
        :param coordinates: A tuple of vectors, each with shape = (-1, 1).
        :type coordinates: tuple[torch.Tensor]
        :return: Function values at sampled points.
        :rtype: torch.Tensor
        """
        if self.enforcer:
            return self.enforcer(net, cond, coordinates)

        if cond.__class__.enforce == BaseCondition.enforce:
            n_params = len(signature(cond.parameterize).parameters)
        else:
            n_params = len(signature(cond.enforce).parameters)
        coordinates = coordinates[:n_params - 1]
        return cond.enforce(net, *coordinates)

    def compute_func_val(self, net, cond, *coordinates):
        r"""Enforce condition on network with inputs. If self.enforcer is set, use it.
        Otherwise, fill cond.enforce() with as many arguments as needed.

        :param net: Network for parameterized solution.
        :type net: torch.nn.Module
        :param cond: Condition (a.k.a. parameterization) for the network.
        :type cond: `neurodiffeq.conditions.BaseCondition`
        :param coordinates: A tuple of vectors, each with shape = (-1, 1).
        :type coordinates: tuple[torch.Tensor]
        :return: Function values at sampled points.
        :rtype: torch.Tensor
        """
        return self._auto_enforce(net, cond, *coordinates)

    def get_solution(self, copy=True, best=True, harmonics_fn=None):
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :param harmonics_fn:
            If set, use it as function basis for returned solution.
        :type harmonics_fn: callable
        :return: The solution after training.
        :rtype: ``neurodiffeq.solvers.BaseSolution``
        """
        nets = self.best_nets if best else self.nets
        conditions = self.conditions
        if copy:
            nets = deepcopy(nets)
            conditions = deepcopy(conditions)

        if harmonics_fn:
            return SolutionSphericalHarmonics(nets, conditions, harmonics_fn=harmonics_fn)
        else:
            return SolutionSpherical(nets, conditions)

    def _get_internal_variables(self):
        available_variables = super(SolverSpherical, self)._get_internal_variables()
        available_variables.update({
            'r_min': self.r_min,
            'r_max': self.r_max,
            'enforcer': self.enforcer,
        })
        return available_variables


class SolutionSpherical(BaseSolution):
    def _compute_u(self, net, condition, rs, thetas, phis):
        return condition.enforce(net, rs, thetas, phis)


class SolutionSphericalHarmonics(SolutionSpherical):

    def __init__(self, nets, conditions, max_degree=None, harmonics_fn=None):
        super(SolutionSphericalHarmonics, self).__init__(nets, conditions)
        if (harmonics_fn is None) and (max_degree is None):
            raise ValueError("harmonics_fn should be specified")

        if max_degree is not None:
            warnings.warn(
                "`max_degree` is DEPRECATED; pass `harmonics_fn` instead, which takes precedence",
                FutureWarning,
            )
            self.harmonics_fn = RealSphericalHarmonics(max_degree=max_degree)

        if harmonics_fn is not None:
            self.harmonics_fn = harmonics_fn

    def _compute_u(self, net, condition, rs, thetas, phis):
        products = condition.enforce(net, rs) * self.harmonics_fn(thetas, phis)
        return torch.sum(products, dim=1)


class Solution1D(BaseSolution):
    def _compute_u(self, net, condition, ts):
        return condition.enforce(net, ts)


class Solver1D(BaseSolver):

    def __init__(self, ode_system, conditions, t_min=None, t_max=None,
                 nets=None, train_generator=None, valid_generator=None, analytic_solutions=None, optimizer=None,
                 loss_fn=None, n_batches_train=1, n_batches_valid=4, metrics=None, n_output_units=1,
                 batch_size=None, shuffle=None):

        if train_generator is None or valid_generator is None:
            if t_min is None or t_max is None:
                raise ValueError(f"Either generator is not provided, t_min and t_max should be both provided: \n"
                                 f"got t_min={t_min}, t_max={t_max}, "
                                 f"train_generator={train_generator}, valid_generator={valid_generator}")

        if train_generator is None:
            train_generator = Generator1D(32, t_min=t_min, t_max=t_max, method='equally-spaced-noisy')
        if valid_generator is None:
            valid_generator = Generator1D(32, t_min=t_min, t_max=t_max, method='equally-spaced')

        self.t_min, self.t_max = t_min, t_max

        super(Solver1D, self).__init__(
            diff_eqs=ode_system,
            conditions=conditions,
            nets=nets,
            train_generator=train_generator,
            valid_generator=valid_generator,
            analytic_solutions=analytic_solutions,
            optimizer=optimizer,
            loss_fn=loss_fn,
            n_batches_train=n_batches_train,
            n_batches_valid=n_batches_valid,
            metrics=metrics,
            n_input_units=1,
            n_output_units=n_output_units,
            shuffle=shuffle,
            batch_size=batch_size,
        )

    def get_solution(self, copy=True, best=True):
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :return:
            A solution object which can be called.
            To evaluate the solution on certain points,
            you should pass the coordinates vector(s) to the returned solution.
        :rtype: BaseSolution
        """
        nets = self.best_nets if best else self.nets
        conditions = self.conditions
        if copy:
            nets = deepcopy(nets)
            conditions = deepcopy(conditions)

        return Solution1D(nets, conditions)

    def _get_internal_variables(self):
        available_variables = super(Solver1D, self)._get_internal_variables()
        available_variables.update({
            't_min': self.t_min,
            't_max': self.t_max,
        })
        return available_variables


class BundleSolution1D(BaseSolution):
    def _compute_u(self, net, condition, *ts):
        return condition.enforce(net, *ts)


class BundleSolver1D(BaseSolver):

    def __init__(self, ode_system, conditions, t_min, t_max,
                 theta_min=None, theta_max=None, eq_param_index=(),
                 nets=None, train_generator=None, valid_generator=None, analytic_solutions=None, optimizer=None,
                 loss_fn=None, n_batches_train=1, n_batches_valid=4, metrics=None, n_output_units=1,
                 batch_size=None, shuffle=None):

        if train_generator is None or valid_generator is None:
            if t_min is None or t_max is None:
                raise ValueError(f"Either generator is not provided, t_min and t_max should be both provided: \n"
                                 f"got t_min={t_min}, t_max={t_max}, "
                                 f"train_generator={train_generator}, valid_generator={valid_generator}")

        if isinstance(theta_min, (float, int)):
            theta_min = (theta_min,)
        elif theta_min is None:
            theta_min = ()

        if isinstance(theta_max, (float, int)):
            theta_max = (theta_max,)
        elif theta_max is None:
            theta_max = ()

        if len(theta_min) != len(theta_max):
            raise ValueError(
                f"length of theta_min and theta_max must be equal, " f"got {len(theta_min)} != {len(theta_max)}"
            )

        r_min = (t_min,) + tuple(theta_min)
        r_max = (t_max,) + tuple(theta_max)

        n_input_units = len(r_min)

        if train_generator is None:
            train_generator = Generator1D(32, t_min=t_min, t_max=t_max, method='equally-spaced-noisy')
            for i in range(n_input_units - 1):
                train_generator ^= Generator1D(32, t_min=r_min[i + 1], t_max=r_max[i + 1],
                                               method='equally-spaced-noisy')
        if valid_generator is None:
            valid_generator = Generator1D(32, t_min=t_min, t_max=t_max, method='equally-spaced')
            for i in range(n_input_units - 1):
                valid_generator ^= Generator1D(32, t_min=r_min[i + 1], t_max=r_max[i + 1], method='equally-spaced')

        self.r_min, self.r_max = r_min, r_max

        N_FUNCTIONS = len(conditions)
        N_COORDS = 1

        eq_param_index = tuple(N_FUNCTIONS + N_COORDS + idx for idx in eq_param_index)
        self.eq_param_index = eq_param_index


        def _diff_eqs_wrapper(*variables):
            pass

        super(BundleSolver1D, self).__init__(
            diff_eqs=_diff_eqs_wrapper,
            conditions=conditions,
            nets=nets,
            train_generator=train_generator,
            valid_generator=valid_generator,
            analytic_solutions=analytic_solutions,
            optimizer=optimizer,
            loss_fn=loss_fn,
            n_batches_train=n_batches_train,
            n_batches_valid=n_batches_valid,
            metrics=metrics,
            n_input_units=n_input_units,
            n_output_units=n_output_units,
            shuffle=shuffle,
            batch_size=batch_size,
        )

    def get_solution(self, copy=True, best=True):
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :return:
            A solution object which can be called.
            To evaluate the solution on certain points,
            you should pass the coordinates vector(s) to the returned solution.
        :rtype: BaseSolution
        """
        nets = self.best_nets if best else self.nets
        conditions = self.conditions
        if copy:
            nets = deepcopy(nets)
            conditions = deepcopy(conditions)

        return BundleSolution1D(nets, conditions)

    def _get_internal_variables(self):
        available_variables = super(BundleSolver1D, self)._get_internal_variables()
        available_variables.update({
            'r_min': self.r_min,
            'r_max': self.r_max,
            'eq_param_index': self.eq_param_index,
        })
        return available_variables


class Solution2D(BaseSolution):
    def _compute_u(self, net, condition, xs, ys):
        return condition.enforce(net, xs, ys)


class Solver2D(BaseSolver):

    def __init__(self, pde_system, conditions, xy_min=None, xy_max=None,
                 nets=None, train_generator=None, valid_generator=None, analytic_solutions=None, optimizer=None,
                 loss_fn=None, n_batches_train=1, n_batches_valid=4, metrics=None, n_output_units=1,
                 batch_size=None, shuffle=None):

        if train_generator is None or valid_generator is None:
            if xy_min is None or xy_max is None:
                raise ValueError(f"Either generator is not provided, xy_min and xy_max should be both provided: \n"
                                 f"got xy_min={xy_min}, xy_max={xy_max}, "
                                 f"train_generator={train_generator}, valid_generator={valid_generator}")

        if train_generator is None:
            train_generator = Generator2D((32, 32), xy_min=xy_min, xy_max=xy_max, method='equally-spaced-noisy')
        if valid_generator is None:
            valid_generator = Generator2D((32, 32), xy_min=xy_min, xy_max=xy_max, method='equally-spaced')

        self.xy_min, self.xy_max = xy_min, xy_max

        super(Solver2D, self).__init__(
            diff_eqs=pde_system,
            conditions=conditions,
            nets=nets,
            train_generator=train_generator,
            valid_generator=valid_generator,
            analytic_solutions=analytic_solutions,
            optimizer=optimizer,
            loss_fn=loss_fn,
            n_batches_train=n_batches_train,
            n_batches_valid=n_batches_valid,
            metrics=metrics,
            n_input_units=2,
            n_output_units=n_output_units,
            shuffle=shuffle,
            batch_size=batch_size,
        )

    def get_solution(self, copy=True, best=True):
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :return:
            A solution object which can be called.
            To evaluate the solution on certain points,
            you should pass the coordinates vector(s) to the returned solution.
        :rtype: BaseSolution
        """
        nets = self.best_nets if best else self.nets
        conditions = self.conditions
        if copy:
            nets = deepcopy(nets)
            conditions = deepcopy(conditions)

        return Solution2D(nets, conditions)

    def _get_internal_variables(self):
        available_variables = super(Solver2D, self)._get_internal_variables()
        available_variables.update({
            'xy_min': self.xy_min,
            'xy_max': self.xy_max,
        })
        return available_variables
