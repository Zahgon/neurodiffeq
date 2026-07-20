import torch
import torch.nn as nn
from warnings import warn


class FCNN(nn.Module):

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None, n_hidden_layers=None,
                 actv=nn.Tanh, hidden_units=None):
        r"""Initializer method.
        """
        super(FCNN, self).__init__()

        if n_hidden_units is None and n_hidden_layers is not None:
            n_hidden_units = 32
        elif n_hidden_units is not None and n_hidden_layers is None:
            n_hidden_layers = 1

        if n_hidden_units is not None or n_hidden_layers is not None:
            if hidden_units is None:
                hidden_units = tuple(n_hidden_units for _ in range(n_hidden_layers + 1))
                warn(f"`n_hidden_units` and `n_hidden_layers` are deprecated, "
                     f"pass `hidden_units={hidden_units}` instead",
                     FutureWarning)
            else:
                warn(f"Ignoring `n_hidden_units` and `n_hidden_layers` in favor of `hidden_units={hidden_units}`",
                     FutureWarning)

        if hidden_units is None:
            hidden_units = (32, 32)

        if not isinstance(hidden_units, tuple):
            hidden_units = tuple(hidden_units)

        units = (n_input_units,) + hidden_units
        layers = []
        for i in range(len(units) - 1):
            layers.append(nn.Linear(units[i], units[i + 1]))
            layers.append(actv())
        layers.append(nn.Linear(units[-1], n_output_units))
        self.NN = torch.nn.Sequential(*layers)

    def forward(self, t):
        pass


class Resnet(nn.Module):

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None, n_hidden_layers=None, actv=nn.Tanh,
                 hidden_units=(32, 32)):
        super(Resnet, self).__init__()

        self.residual = FCNN(
            n_input_units=n_input_units,
            n_output_units=n_output_units,
            n_hidden_units=n_hidden_units,
            n_hidden_layers=n_hidden_layers,
            actv=actv,
            hidden_units=hidden_units,
        )
        self.skip_connection = nn.Linear(n_input_units, n_output_units, bias=False)

    def forward(self, t):
        pass


class MonomialNN(nn.Module):

    def __init__(self, degrees):
        super(MonomialNN, self).__init__()

        if isinstance(degrees, int):
            degrees = [d for d in range(1, degrees + 1)]
        self.degrees = tuple(degrees)

        if len(self.degrees) == 0:
            raise ValueError(f"No degrees used, check `degrees` argument again")
        if 0 in degrees:
            warn("One of the degrees is 0 which might introduce redundant features")
        if len(set(self.degrees)) < len(self.degrees):
            warn(f"Duplicate degrees found: {self.degrees}")

    def forward(self, x):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(degrees={self.degrees})"

    def __str__(self):
        return self.__repr__()


class SinActv(nn.Module):

    def __init__(self):
        """Initializer method.
        """
        super().__init__()

    def forward(self, input_):
        pass


class Swish(nn.Module):

    def __init__(self, beta=1.0, trainable=False):
        super(Swish, self).__init__()

        beta = float(beta)
        self.trainable = trainable
        if trainable:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.beta = beta

    def forward(self, x):
        pass

class APTx(nn.Module):

    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, trainable=False):
        super(APTx, self).__init__()
        alpha = float(alpha)
        beta = float(beta)
        gamma = float(gamma)
        self.trainable = trainable
        if self.trainable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma

    def forward(self, x):
        pass
