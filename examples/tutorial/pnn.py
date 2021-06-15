"""
Implementation of a Probabilistic Neural Net.
"""
import abc
from collections import OrderedDict
from copy import deepcopy
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def train_simple_pnn(tr_x, tr_y, val_x, val_y, epochs):
    """Train a simple PNN.
    Args:
        tr_x: The training x data as a numpy array.
        tr_y: The training y data as a numpy array.
        val_x: The validation x data as a numpy array.
        val_y: The validation y data as a numpy array.
        epochs: Number of epochs to train for.
    """
    pnn = PNN(
            input_dim=tr_x.shape[1],
            output_dim=tr_y.shape[1],
            encoder_hidden_sizes=[64, 64],
            mean_hidden_sizes=[],
            logvar_hidden_sizes=[],
            latent_dim=64,
    )
    tr_data = DataLoader(
        TensorDataset(torch.Tensor(tr_x), torch.Tensor(tr_y)),
        batch_size=256,
        shuffle=True,
    )
    val_data = DataLoader(
        TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y)),
        batch_size=256,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(pnn.parameters(), lr=1e-3)
    pbar = tqdm(total=epochs)
    best_val_loss = float('inf')
    best_state = None
    for _ in range(epochs):
        epoch_tr_loss = 0
        for batch_x, batch_y in tr_data:
            loss = get_loss(pnn, batch_x, batch_y)
            optimizer.zero_grad();
            loss.backward()
            optimizer.step()
            epoch_tr_loss += loss.item()
        epoch_tr_loss /= len(tr_data)
        epoch_val_loss = 0
        for batch_x, batch_y in val_data:
            with torch.no_grad():
                epoch_val_loss += get_loss(pnn, batch_x, batch_y).item()
        epoch_val_loss /= len(val_data)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = deepcopy(pnn.state_dict())
        pbar.set_postfix(ordered_dict=OrderedDict(
            TrainLoss=epoch_tr_loss,
            ValLoss=epoch_val_loss,
            BestValLoss=best_val_loss,
        ))
        pbar.update(1)
    pnn.load_state_dict(best_state)
    return pnn

def get_loss(pnn, x_in, labels):
    mean, logvar = pnn.forward(x_in)
    sqdiffs = (mean - labels) ** 2
    return torch.mean(torch.exp(-logvar) * sqdiffs + logvar)

class MLP(torch.nn.Module):
    """MLP Network."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        output_dim: int,
        hidden_activation=torch.nn.functional.relu,
        output_activation=None,
        linear_wrapper=None,
    ):
        """Constructor."""
        super(MLP, self).__init__()
        self._linear_wrapper = linear_wrapper
        if len(hidden_sizes) == 0:
            self._add_linear_layer(input_dim, output_dim, 0)
            self.n_layers = 1
        else:
            self._add_linear_layer(input_dim, hidden_sizes[0], 0)
            for hidx in range(len(hidden_sizes) - 1):
                self._add_linear_layer(hidden_sizes[hidx],
                                       hidden_sizes[hidx+1], hidx + 1)
            self._add_linear_layer(hidden_sizes[-1], output_dim,
                                   len(hidden_sizes))
            self.n_layers = len(hidden_sizes) + 1
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def forward(
            self,
            net_in: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through network."""
        curr = net_in
        for layer_num in range(self.n_layers - 1):
            curr = getattr(self, 'linear_%d' % layer_num)(curr)
            curr = self.hidden_activation(curr)
        curr = getattr(self, 'linear_%d' % (self.n_layers - 1))(curr)
        if self.output_activation is not None:
            return self.output_activation(curr)
        return curr

    def _add_linear_layer(self, lin_in, lin_out, layer_num):
        layer = torch.nn.Linear(lin_in, lin_out)
        if self._linear_wrapper is not None:
            layer = self._linear_wrapper(layer)
        self.add_module('linear_%d' % layer_num, layer)


class PNN(torch.nn.Module):
    """Network that returns mean and logvar of a Gaussian."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        encoder_hidden_sizes: Sequence[int],
        mean_hidden_sizes: Sequence[int],
        logvar_hidden_sizes: Sequence[int],
        latent_dim: int,
        hidden_activation=torch.nn.functional.relu,
        linear_wrappers = [None, None, None],
    ):
        """Constructor."""
        super(PNN, self).__init__()
        self.encode_net = MLP(
            input_dim=input_dim,
            hidden_sizes=encoder_hidden_sizes,
            output_dim=latent_dim,
            hidden_activation=hidden_activation,
            output_activation=hidden_activation,
            linear_wrapper=linear_wrappers[0],
        )
        self.mean_net = MLP(
            input_dim=latent_dim,
            hidden_sizes=mean_hidden_sizes,
            output_dim=output_dim,
            hidden_activation=hidden_activation,
            linear_wrapper=linear_wrappers[1],
        )
        self.logvar_net = MLP(
            input_dim=latent_dim,
            hidden_sizes=logvar_hidden_sizes,
            output_dim=output_dim,
            hidden_activation=hidden_activation,
            linear_wrapper=linear_wrappers[2],
        )

    def forward(
            self,
            net_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network. Returns (mean, logvar)"""
        latent = self.encode_net.forward(net_in)
        return self.mean_net.forward(latent), self.logvar_net.forward(latent)

    def get_mean_and_standard_deviation(
            self,
            x_input: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the mean and standard deviation as numpy ndarrays."""
        torch_input = torch.Tensor(x_input)
        with torch.no_grad():
            mean, logvar = self.forward(torch_input)
        mean = mean.numpy().flatten()
        std = (0.5 * logvar).exp().numpy().flatten()
        return mean, std
