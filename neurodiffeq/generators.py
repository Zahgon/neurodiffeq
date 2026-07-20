import torch
import numpy as np
from typing import List


def _chebyshev_first(a, b, n):
    nodes = torch.cos(((torch.arange(n) + 0.5) / n) * np.pi)
    nodes = ((a + b) + (b - a) * nodes) / 2
    nodes.requires_grad_(True)
    return nodes


def _chebyshev_second(a, b, n):
    nodes = torch.cos(torch.arange(n) / float(n - 1) * np.pi)
    nodes = ((a + b) + (b - a) * nodes) / 2
    nodes.requires_grad_(True)
    return nodes

def _chebyshev_second_noisy(a, b, n):
    pass


def _latin_hypercube(a, b, n):
    pass


def _compute_log_negative(t_min, t_max, whence):
    pass


class BaseGenerator:

    def __init__(self):
        self.size = None

    def get_examples(self) -> List[torch.Tensor]:
        pass  # pragma: no cover

    @staticmethod
    def check_generator(obj):
        pass

    def __add__(self, other):
        self.check_generator(other)
        return ConcatGenerator(self, other)

    def __mul__(self, other):
        self.check_generator(other)
        return EnsembleGenerator(self, other)

    def __xor__(self, other):
        self.check_generator(other)
        return MeshGenerator(self, other)

    def _internal_vars(self) -> dict:
        pass

    @staticmethod
    def _obj_repr(obj) -> str:
        pass

    def __repr__(self):
        d = self._internal_vars()
        keys = ', '.join(f'{k}={self._obj_repr(d[k])}' for k in d)
        return f'{self.__class__.__name__}({keys})'


class Generator1D(BaseGenerator):

    def __init__(self, size, t_min=0.0, t_max=1.0, method='uniform', noise_std=None):
        r"""Initializer method

        .. note::
            A instance method `get_examples` is dynamically created to generate 1-D training points.
            It will be called by the function `solve` and `solve_system`.
        """
        super(Generator1D, self).__init__()
        self.size = size
        self.t_min, self.t_max = t_min, t_max
        self.method = method
        if noise_std:
            self.noise_std = noise_std
        else:
            self.noise_std = ((t_max - t_min) / size) / 4.0
        if method == 'uniform':
            self.examples = torch.zeros(self.size, requires_grad=True)
            self.getter = lambda: self.examples + torch.rand(self.size) * (self.t_max - self.t_min) + self.t_min
        elif method == 'equally-spaced':
            self.examples = torch.linspace(self.t_min, self.t_max, self.size, requires_grad=True)
            self.getter = lambda: self.examples
        elif method == 'equally-spaced-noisy':
            self.examples = torch.linspace(self.t_min, self.t_max, self.size, requires_grad=True)
            self.getter = lambda: torch.normal(mean=self.examples, std=self.noise_std)
        elif method == 'log-spaced':
            start, end = _compute_log_negative(t_min, t_max, self.__class__)
            self.examples = torch.logspace(start, end, self.size, requires_grad=True)
            self.getter = lambda: self.examples
        elif method == 'log-spaced-noisy':
            start, end = _compute_log_negative(t_min, t_max, self.__class__)
            self.examples = torch.logspace(start, end, self.size, requires_grad=True)
            self.getter = lambda: torch.normal(mean=self.examples, std=self.noise_std)
        elif method in ['chebyshev', 'chebyshev1']:
            self.examples = _chebyshev_first(t_min, t_max, size)
            self.getter = lambda: self.examples
        elif method == 'chebyshev2':
            self.examples = _chebyshev_second(t_min, t_max, size)
            self.getter = lambda: self.examples
        elif method == 'chebyshev2-noisy':
            self.getter = lambda: _chebyshev_second_noisy(t_min, t_max, size)
        elif method == 'latin-hypercube':
            self.getter = lambda: _latin_hypercube(t_min, t_max, size)
        else:
            raise ValueError(f'Unknown method: {method}')

    def get_examples(self):
        return self.getter()

    def _internal_vars(self):
        pass


class Generator2D(BaseGenerator):

    def __init__(self, grid=(10, 10), xy_min=(0.0, 0.0), xy_max=(1.0, 1.0), method='equally-spaced-noisy',
                 xy_noise_std=None):
        r"""Initializer method

        .. note::
            A instance method `get_examples` is dynamically created to generate 2-D training points.
            It will be called by the function `solve2D`.
        """
        super(Generator2D, self).__init__()
        self.grid = grid
        self.size = grid[0] * grid[1]
        self.xy_min = xy_min
        self.xy_max = xy_max
        self.method = method
        self.xy_noise_std = xy_noise_std

        if method == 'equally-spaced':
            x = torch.linspace(xy_min[0], xy_max[0], grid[0], requires_grad=True)
            y = torch.linspace(xy_min[1], xy_max[1], grid[1], requires_grad=True)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            self.grid_x, self.grid_y = grid_x.flatten(), grid_y.flatten()
            self.getter = lambda: (self.grid_x, self.grid_y)
        elif method == 'equally-spaced-noisy':
            x = torch.linspace(xy_min[0], xy_max[0], grid[0], requires_grad=True)
            y = torch.linspace(xy_min[1], xy_max[1], grid[1], requires_grad=True)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            self.grid_x, self.grid_y = grid_x.flatten(), grid_y.flatten()
            if xy_noise_std:
                self.noise_xstd, self.noise_ystd = xy_noise_std
            else:
                self.noise_xstd = ((xy_max[0] - xy_min[0]) / grid[0]) / 4.0
                self.noise_ystd = ((xy_max[1] - xy_min[1]) / grid[1]) / 4.0
            self.getter = lambda: (
                torch.normal(mean=self.grid_x, std=self.noise_xstd),
                torch.normal(mean=self.grid_y, std=self.noise_ystd)
            )
        elif method in ['chebyshev1', 'chebyshev']:
            x = _chebyshev_first(xy_min[0], xy_max[0], grid[0])
            y = _chebyshev_first(xy_min[1], xy_max[1], grid[1])
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            self.grid_x, self.grid_y = grid_x.flatten(), grid_y.flatten()
            self.getter = lambda: (self.grid_x, self.grid_y)
        elif method == 'chebyshev2':
            x = _chebyshev_second(xy_min[0], xy_max[0], grid[0])
            y = _chebyshev_second(xy_min[1], xy_max[1], grid[1])
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            self.grid_x, self.grid_y = grid_x.flatten(), grid_y.flatten()
            self.getter = lambda: (self.grid_x, self.grid_y)
        elif method == 'chebyshev2-noisy':
            self.getter = self.generate(xy_min, xy_max, grid, method='chebyshev2-noisy')
        elif method == 'latin-hypercube':
            x = _latin_hypercube(xy_min[0], xy_max[0], grid[0])
            y = _latin_hypercube(xy_min[1], xy_max[1], grid[1])
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            self.grid_x, self.grid_y = grid_x.flatten(), grid_y.flatten()
            self.getter = lambda: (self.grid_x, self.grid_y)
        else:
            raise ValueError(f'Unknown method: {method}')

    def generate(self, xy_min, xy_max, grid, method='chebyshev2-noisy'):
        pass

    def get_examples(self):
        return self.getter()

    def _internal_vars(self) -> dict:
        pass


class Generator3D(BaseGenerator):

    def __init__(self, grid=(10, 10, 10), xyz_min=(0.0, 0.0, 0.0), xyz_max=(1.0, 1.0, 1.0),
                 method='equally-spaced-noisy'):
        r"""Initializer method

        .. note::
            A instance method `get_examples` is dynamically created to generate 2-D training points.
            It will be called by the function `solve2D`.
        """
        super(Generator3D, self).__init__()
        self.size = grid[0] * grid[1] * grid[2]
        self.grid = grid
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.method = method

        if method in ['equally-spaced', 'equally-spaced-noisy']:
            x = torch.linspace(xyz_min[0], xyz_max[0], grid[0], requires_grad=True)
            y = torch.linspace(xyz_min[1], xyz_max[1], grid[1], requires_grad=True)
            z = torch.linspace(xyz_min[2], xyz_max[2], grid[2], requires_grad=True)
        elif method in ['chebyshev', 'chebyshev1']:
            x = _chebyshev_first(xyz_min[0], xyz_max[0], grid[0])
            y = _chebyshev_first(xyz_min[1], xyz_max[1], grid[1])
            z = _chebyshev_first(xyz_min[2], xyz_max[2], grid[2])
        elif method == 'chebyshev2':
            x = _chebyshev_second(xyz_min[0], xyz_max[0], grid[0])
            y = _chebyshev_second(xyz_min[1], xyz_max[1], grid[1])
            z = _chebyshev_second(xyz_min[2], xyz_max[2], grid[2])
        elif method == 'latin-hypercube':
            x = _latin_hypercube(xyz_min[0], xyz_max[0], grid[0])
            y = _latin_hypercube(xyz_min[1], xyz_max[1], grid[1])
            z = _latin_hypercube(xyz_min[2], xyz_max[2], grid[2])
        else:
            raise ValueError(f"Unknown method: {method}")

        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        self.grid_x, self.grid_y, self.grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

        if method in ['equally-spaced', 'chebyshev', 'chebyshev1', 'chebyshev2', 'latin-hypercube']:
            self.getter = lambda: (self.grid_x, self.grid_y, self.grid_z)
        elif method == 'equally-spaced-noisy':
            self.noise_xmean = torch.zeros(self.size)
            self.noise_ymean = torch.zeros(self.size)
            self.noise_zmean = torch.zeros(self.size)
            self.noise_xstd = torch.ones(self.size) * ((xyz_max[0] - xyz_min[0]) / grid[0]) / 4.0
            self.noise_ystd = torch.ones(self.size) * ((xyz_max[1] - xyz_min[1]) / grid[1]) / 4.0
            self.noise_zstd = torch.ones(self.size) * ((xyz_max[2] - xyz_min[2]) / grid[2]) / 4.0
            self.getter = lambda: (
                self.grid_x + torch.normal(mean=self.noise_xmean, std=self.noise_xstd),
                self.grid_y + torch.normal(mean=self.noise_ymean, std=self.noise_ystd),
                self.grid_z + torch.normal(mean=self.noise_zmean, std=self.noise_zstd),
            )
        else:
            raise ValueError(f'Unknown method: {method}')

    def get_examples(self):
        return self.getter()

    def _internal_vars(self) -> dict:
        pass


class GeneratorND(BaseGenerator):

    def __init__(self, grid=(10, 10), r_min=(0.0, 0.0), r_max=(1.0, 1.0),
                 methods=['equally-spaced', 'equally-spaced'], noisy=True, r_noise_std=None,
                 **kwargs):

        super(GeneratorND, self).__init__()
        self.size = np.prod(grid)
        self.grid = grid
        self.r_min = r_min
        self.r_max = r_max
        self.methods = methods
        self.noisy = noisy
        self.r_noise_std = r_noise_std

        if isinstance(methods, str):
            methods = [methods]
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(r_min, float) or isinstance(r_min, int):
            r_min = (r_min,)
        if isinstance(r_max, float) or isinstance(r_max, int):
            r_max = (r_max,)
        if isinstance(r_noise_std, float) or isinstance(r_noise_std, int):
            r_noise_std = (r_noise_std,)

        N = len(grid)
        cut = kwargs.pop('cut', tuple((None, None) for i in range(N)))
        base = kwargs.pop('base', tuple(10 for i in range(N)))
        abs_value = kwargs.pop('abs_value', False)

        if kwargs:
            raise ValueError(f'Unknown keyword argument(s): {list(kwargs.keys())}')
        if isinstance(base, float) or isinstance(base, int):
            base = (base,)
        if isinstance(cut[0], float) or isinstance(cut[0], int) or cut[0] is None:
            cut = (cut,)

        r = []
        r_noise_std_list = []
        for i in range(N):

            method = methods[i]

            if r_noise_std:
                noise_rstd = r_noise_std[i]
            else:
                noise_rstd = ((r_max[i] - r_min[i]) / grid[i]) / 4.0

            if method == 'equally-spaced':
                x = torch.linspace(r_min[i], r_max[i], grid[i], requires_grad=True)
                noise_rstd_tensor = noise_rstd * torch.ones(x.size())
            elif method == 'uniform':
                x = torch.zeros(grid[i], requires_grad=True)
                x = x + torch.rand(grid[i]) * (r_max[i] - r_min[i]) + r_min[i]
                noise_rstd_tensor = torch.zeros(x.size())
            elif method == 'log-spaced':
                r_min_log = np.log10(r_min[i])
                r_max_log = np.log10(r_max[i])
                x = torch.logspace(r_min_log, r_max_log, grid[i], requires_grad=True)
                noise_rstd_tensor = (noise_rstd * torch.logspace(r_min_log, r_max_log, grid[i]))
            elif method == 'exp-spaced':
                r_min_exp = base[i] ** r_min[i]
                r_max_exp = base[i] ** r_max[i]
                x = torch.linspace(r_min_exp, r_max_exp, grid[i], requires_grad=True)
                x = (torch.log(x) / np.log(base[i])).clone().detach().requires_grad_(True)
                noise_rstd_tensor = (noise_rstd * x).clone().detach()
            elif method in ['chebyshev', 'chebyshev1']:
                x = _chebyshev_first(r_min[i], r_max[i], grid[i])
                noise_rstd_tensor = noise_rstd * torch.ones(x.size())
            elif method == 'chebyshev2':
                x = _chebyshev_second(r_min[i], r_max[i], grid[i])
                noise_rstd_tensor = noise_rstd * torch.ones(x.size())
            else:
                raise ValueError(f'Unknown method: {method}')

            x = x[cut[i][0]:cut[i][1]]
            noise_rstd_tensor = noise_rstd_tensor[cut[i][0]:cut[i][1]]
            r.append(x)
            r_noise_std_list.append(noise_rstd_tensor)

        grid_r = torch.meshgrid(r, indexing='ij')
        grid_std = torch.meshgrid(r_noise_std_list, indexing='ij')
        self.grid_r = [grid_r[j].flatten() for j in range(N)]
        self.grid_std = [grid_std[j].flatten() for j in range(N)]
        if noisy:
            if abs_value:
                self.getter = lambda: tuple(torch.abs(torch.normal(self.grid_r[n], self.grid_std[n])) for n in range(N))
            else:
                self.getter = lambda: tuple(torch.normal(self.grid_r[n], self.grid_std[n]) for n in range(N))
        else:
            self.getter = lambda: tuple(self.grid_r[n] for n in range(N))

    def get_examples(self):
        return self.getter()

    def _internal_vars(self) -> dict:
        pass


class GeneratorSpherical(BaseGenerator):

    def __init__(self, size, r_min=0., r_max=1., method='equally-spaced-noisy'):
        super(GeneratorSpherical, self).__init__()
        if r_min < 0 or r_max < r_min:
            raise ValueError(f"Illegal range [{r_min}, {r_max}]")

        if method == 'equally-spaced-noisy':
            lower = r_min ** 2
            upper = r_max ** 2
            rng = upper - lower
            self.get_r = lambda: torch.sqrt(rng * torch.rand(self.shape) + lower)
        elif method == "equally-radius-noisy":
            lower = r_min
            upper = r_max
            rng = upper - lower
            self.get_r = lambda: rng * torch.rand(self.shape) + lower
        else:
            raise ValueError(f'Unknown method: {method}')

        self.size = size  # stored for `solve_spherical_system` to access
        self.r_min = r_min
        self.r_max = r_max
        self.method = method
        self.shape = (size,)  # used for `self.get_example()`

    def get_examples(self):
        a = torch.rand(self.shape)
        b = torch.rand(self.shape)
        c = torch.rand(self.shape)
        denom = a + b + c
        epsilon = 1e-6
        x = torch.sqrt(a / denom) + epsilon
        y = torch.sqrt(b / denom) + epsilon
        z = torch.sqrt(c / denom) + epsilon
        sign_x = torch.randint(0, 2, self.shape, dtype=x.dtype) * 2 - 1
        sign_y = torch.randint(0, 2, self.shape, dtype=y.dtype) * 2 - 1
        sign_z = torch.randint(0, 2, self.shape, dtype=z.dtype) * 2 - 1

        x = x * sign_x
        y = y * sign_y
        z = z * sign_z

        theta = torch.acos(z).requires_grad_(True)
        phi = -torch.atan2(y, x) + np.pi  # atan2 ranges (-pi, pi] instead of [0, 2pi)
        phi.requires_grad_(True)
        r = self.get_r().requires_grad_(True)

        return r, theta, phi

    def _internal_vars(self) -> dict:
        pass


class ConcatGenerator(BaseGenerator):

    def __init__(self, *generators):
        super(ConcatGenerator, self).__init__()
        self.generators = generators
        self.size = sum(gen.size for gen in generators)

    def get_examples(self):
        all_examples = [gen.get_examples() for gen in self.generators]
        if isinstance(all_examples[0], torch.Tensor):
            return torch.cat(all_examples)
        segmented = zip(*all_examples)
        return [torch.cat(seg) for seg in segmented]

    def _internal_vars(self) -> dict:
        pass


class StaticGenerator(BaseGenerator):

    def __init__(self, generator):
        super(StaticGenerator, self).__init__()
        self.generator = generator
        self.size = generator.size
        self.examples = generator.get_examples()

    def get_examples(self):
        return self.examples

    def _internal_vars(self) -> dict:
        pass


class PredefinedGenerator(BaseGenerator):

    def __init__(self, *xs):
        super(PredefinedGenerator, self).__init__()
        self.size = len(xs[0])
        for x in xs:
            if self.size != len(x):
                raise ValueError('tensors of different lengths encountered {self.size} != {len(x)}')
        xs = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in xs]
        self.xs = [torch.flatten(x).requires_grad_(True) for x in xs]

        if len(self.xs) == 1:
            self.xs = self.xs[0]

    def get_examples(self):
        """Returns the training points. Points are fixed and predefined.

            :returns: The predefined training points
            :rtype: tuple[`torch.Tensor`]
        """
        return self.xs

    def _internal_vars(self) -> dict:
        pass


class TransformGenerator(BaseGenerator):

    def __init__(self, generator, transforms=None, transform=None):
        super(TransformGenerator, self).__init__()
        self.generator = generator
        self.size = generator.size
        if transforms is not None and transform is not None:
            raise ValueError("transform and transforms cannot be both specified")
        if transforms is not None:
            self.trans = [
                (lambda x: x) if t is None else t
                for t in transforms
            ]
        elif transform is not None:
            self.trans = transform
        else:
            self.trans = lambda x: x

    def get_examples(self):
        xs = self.generator.get_examples()
        if isinstance(xs, torch.Tensor):
            if callable(self.trans):
                return self.trans(xs)
            else:
                return self.trans[0](xs)
        if callable(self.trans):
            return self.trans(*xs)
        else:
            return tuple(t(x) for t, x in zip(self.trans, xs))

    def _internal_vars(self) -> dict:
        pass


class EnsembleGenerator(BaseGenerator):

    def __init__(self, *generators):
        super(EnsembleGenerator, self).__init__()
        self.size = generators[0].size
        for i, gen in enumerate(generators):
            if gen.size != self.size:
                raise ValueError(f"gens[{i}].size ({gen.size}) != gens[0].size ({self.size})")
        self.generators = generators

    def get_examples(self):
        ret = tuple()
        for g in self.generators:
            ex = g.get_examples()
            if isinstance(ex, list):
                ex = tuple(ex)
            elif isinstance(ex, torch.Tensor):
                ex = (ex,)
            ret += ex

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def _internal_vars(self) -> dict:
        pass


class MeshGenerator(BaseGenerator):

    def __init__(self, *generators):
        super(MeshGenerator, self).__init__()
        self.generators = []
        for g in generators:
            if isinstance(g, MeshGenerator):
                for s in g.generators:
                    self.generators.append(s)
            else:
                self.generators.append(g)
        self.size = np.prod(tuple(g.size for g in self.generators))

    def get_examples(self):
        ret = tuple()
        for g in self.generators:
            ex = g.get_examples()
            if isinstance(ex, list):
                ex = tuple(ex)
            elif isinstance(ex, torch.Tensor):
                ex = (ex,)
            ret += ex

        if len(ret) == 1:
            return ret[0]
        else:
            ret = torch.meshgrid(ret, indexing='ij')
            ret_f = tuple()
            for r in ret:
                ret_f += (r.flatten(),)
            return ret_f

    def _internal_vars(self) -> dict:
        pass


class FilterGenerator(BaseGenerator):

    def __init__(self, generator, filter_fn, size=None, update_size=True):
        super(FilterGenerator, self).__init__()
        self.generator = generator
        self.filter_fn = filter_fn
        if size is None:
            self.size = generator.size
        else:
            self.size = size
        self.update_size = update_size

    def get_examples(self):
        xs = self.generator.get_examples()
        if isinstance(xs, torch.Tensor):
            xs = [xs]
        mask = self.filter_fn(xs)
        xs = [x[mask] for x in xs]
        if self.update_size:
            self.size = len(xs[0])
        if len(xs) == 1:
            return xs[0]
        else:
            return xs

    def _internal_vars(self) -> dict:
        pass


class ResampleGenerator(BaseGenerator):

    def __init__(self, generator, size=None, replacement=False):
        super(ResampleGenerator, self).__init__()
        self.generator = generator
        if size is None:
            self.size = generator.size
        else:
            self.size = size
        self.replacement = replacement

    def get_examples(self):
        if self.replacement:
            indices = torch.randint(self.generator.size, (self.size,))
        else:
            indices = torch.randperm(self.generator.size)[:self.size]

        xs = self.generator.get_examples()
        if isinstance(xs, torch.Tensor):
            return xs[indices]
        else:
            return [x[indices] for x in xs]

    def _internal_vars(self) -> dict:
        pass


class BatchGenerator(BaseGenerator):

    def __init__(self, generator, batch_size):
        super(BatchGenerator, self).__init__()

        if generator.size <= 0:
            raise ValueError(f"generator has size {generator.size} <= 0")
        self.generator = generator
        self.size = batch_size
        self.cached_xs = self.generator.get_examples()
        if isinstance(self.cached_xs, torch.Tensor):
            self.cached_xs = [self.cached_xs]
        if isinstance(self.cached_xs, tuple):
            self.cached_xs = list(self.cached_xs)

    def get_examples(self):
        while len(self.cached_xs[0]) < self.size:
            new = self.generator.get_examples()
            if isinstance(new, torch.Tensor):
                new = [new]
            self.cached_xs = [torch.cat([x, n]) for x, n in zip(self.cached_xs, new)]

        batch = [x[:self.size] for x in self.cached_xs]
        self.cached_xs = [x[self.size:] for x in self.cached_xs]

        if len(batch) == 1:
            return batch[0]
        else:
            return batch

    def _internal_vars(self) -> dict:
        pass


class SamplerGenerator(BaseGenerator):
    def __init__(self, generator):
        super(SamplerGenerator, self).__init__()
        self.generator = generator
        self.size = generator.size

    def get_examples(self) -> List[torch.Tensor]:
        samples = self.generator.get_examples()
        if isinstance(samples, torch.Tensor):
            samples = [samples]
        samples = [u.reshape(-1, 1) for u in samples]
        return samples

    def _internal_vars(self) -> dict:
        pass
