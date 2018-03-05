"""Base model classes."""
from torch import nn
import torch
from torch.autograd import Variable
from ext import layers
import numpy as np


class Model(nn.Module):
    """Base for a model.

    Defines the basic contract any consuming class is expecting:
      * forward(*args): the forward pass of the network. Must return
        predictions, loss, and accuracy.
      * optimize(loss): parameter update step.

    Attributes:
      param_groups: List of dictionaries defining named parameter groups.
    """

    def __init__(self, name, config):
        """Create a new Model.

        Args:
          name: String, unique identifier.
          config: Dictionary of configuration values. These values are flexible
            and model-dependent except for a "name" key we expect to identify
            the model and or training run.
        """
        super(Model, self).__init__()
        # set all config values as attributes on the model for ease of access
        self.config = config
        for key in config.keys():
            setattr(self, key, config[key])
        # override the name with the run number appended name
        self.name = name
        self.param_groups = []

    @staticmethod
    def accuracy(preds, labels):
        """Determine the accuracy of a batch.

        Args:
          preds: torch.autograd.Variable, vector of predictions.
          labels: torch.LongTensor, vector of labels.

        Returns:
          torch.autograd.Variable (Float), scalar.
        """
        correct = preds == labels
        return correct.sum().float() / correct.shape[0]

    def add_param_group(self, params, name, lr=None, l2=None):
        """Add params to the param groups.

        Args:
          params: List of torch.nn.Parameters.
          name: String, a name for the parameter group
          lr: Float. Optional. Learning rate specific to this param group.
          l2: Float. Optional. L2 regularization penalty specific to this group.
        """
        group = {'params': params, 'name': name}
        if lr:
            group['lr'] = lr
        if l2:
            group['l2'] = l2
        self.param_groups.append(group)

    def forward(self, batch):
        """Forward step of the network. Must be implemented by deriving classes.

        Returns:
          predictions
          loss
          accuracy
        """
        raise NotImplementedError

    def optimize(self, loss):
        """Backward step of the network.

        Implemented by default, but deriving classes must define an attribute
        "optimizer" in their __init__ method.
        """
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def n_params(self):
        return sum([np.prod(p.size()) for p in self.parameters()])

    @staticmethod
    def predictions(logits):
        """Get the predictions for a batch from logits.

        Args:
          logits: torch.autograd.Variable, a vector of logits.

        Returns:
          torch.autograd.Variable (Long), vector of predictions.
        """
        return logits.max(1)[1]

    def xavier_uniform(self, weight, activation):
        """Xavier initialization for a parameter matrix.

        Initializes with torch.nn.init.xavier_uniform with the right gain given
        the activation function as defined here:
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
        if activation == 'none':
            nn.init.xavier_uniform(weight, gain=layers.gain(activation))
        elif activation == 'sigmoid':
            nn.init.xavier_uniform(weight, gain=layers.gain(activation))
        elif activation == 'tanh':
            nn.init.xavier_uniform(weight, gain=layers.gain(activation))
        elif activation == 'relu':
            nn.init.xavier_uniform(weight, gain=layers.gain(activation))


class TextModel(Model):
    """Base text model.

    Extends base mode with embedding and lookup.

    Attributes:
      param_groups: List of dictionaries defining named parameter groups.
      embeds: torch.nn.Embedding.
    """

    def __init__(self, name, config, embed_mat):
        """Create a new TextModel.

        Expects config to have a boolean "tune_embeds".

        Raises:
          ValueError: if "tune_embeds" not in config.
        """
        if 'tune_embeds' not in config.keys():
            raise ValueError('config must define "tune_embeds".')
        super(TextModel, self).__init__(name, config)
        self.embeds = nn.Embedding(embed_mat.shape[0], embed_mat.shape[1])
        self.embeds.weight = nn.Parameter(torch.from_numpy(embed_mat),
                                          requires_grad=self.tune_embeds)

    def lookup(self, ixs, rnn=False):
        """Embedding lookup.

        Args:
          ixs: numpy.ndarray of integer vocab indices.
          rnn: Bool, indicating whether we will subsequently us an rnn.

        Returns:
          torch.autograd.Variable (Float), matrix of word vectors.
        """
        if torch.cuda.is_available():
            lookup_tensor = Variable(
                torch.LongTensor(ixs), requires_grad=False).cuda()
        else:
            lookup_tensor = Variable(torch.LongTensor(ixs), requires_grad=False)
        vecs = self.embeds(lookup_tensor)
        if rnn:
            vecs = vecs.permute([1, 0, 2])
        return vecs
