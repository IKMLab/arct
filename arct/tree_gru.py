"""Models for Tree-GRUs."""
import torch
from torch.autograd import Variable
from torch import nn


class BatchTreeGRUCell(nn.Module):
    def __init__(self):
        super(BatchTreeGRUCell, self).__init__()
        self.

