"""Reusable neural net layers."""
import torch
from torch import nn
import numpy as np


def activation_fn(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'sigmoid':
        return torch.sigmoid
    elif activation == 'tanh':
        return torch.tanh
    else:
        raise ValueError('Unrecognized activation "%r"' % activation)


def gain(activation):
    """Xavier initialization gains.

    https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py

    ============ ==========================================
    nonlinearity gain
    ============ ==========================================
    linear       :math:`1`
    conv{1,2,3}d :math:`1`
    sigmoid      :math:`1`
    tanh         :math:`5 / 3`
    relu         :math:`\sqrt{2}`
    leaky_relu   :math:`\sqrt{2 / (1 + negative\_slope^2)}`
    ============ ==========================================

    Args:
      weight: torch.autograd.Variable.
      activation: String in [none, sig, tanh, relu].

    Raises:
      ValueError: if activation not recognized.
    """
    if activation == 'relu':
        return np.sqrt(2)
    elif activation == 'sigmoid':
        return 1.
    elif activation == 'tanh':
        return 5 / 3.
    elif activation == 'none':
        return 1.
    else:
        raise ValueError('Unrecognized activation "%r"' % activation)


class MLP(nn.Module):

    def __init__(self, input_size, config):
        super(MLP, self).__init__()
        seq = []
        activation = activation_fn(config['mlp_act'])
        for i in range(1, config['mlp_layers'] + 1):
            layer = nn.Linear(
                in_features=input_size if i == 1 else config['mlp_size'],
                out_features=config['mlp_size'])
            setattr(self, 'layer_%s' % i, layer)
            nn.init.xavier_uniform(layer.weight, gain(config['mlp_act']))
            if torch.cuda.is_available():
                layer.cuda()
            seq.append(layer)
            seq.append(activation)
            seq.append(nn.Dropout(config['p_drop_mlp']))
        self.mlp = nn.Sequential(*seq)

    def forward(self, features):
        return self.mlp(features)


class MaxPooling:
    """Max-pooling."""

    def __init__(self, dim=0):
        """Create a new MaxPooling.

        Args:
          dim: Integer, the dimension across which to reduce.
        """
        self.dim = dim

    def __call__(self, tensor):
        """Perform max-pooling across time.

        Args:
          tensor: torch.Tensor or deriving class.

        Returns:
          torch.Tensor or deriving class.
        """
        return torch.max(tensor, self.dim)[0]

