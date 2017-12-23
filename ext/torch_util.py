"""Utility functions for PyTorch."""
import numpy as np
from ext import vocab
import torch
from torch.autograd import Variable
from torch import nn


class CollateForRNN:
    """Wrapping functions for preparing text for PyTorch RNN processing.

    This is specifically designed for the collate function - not use in the
    model. The reason for this separation is multithreading, and a cuda error
    received when I tried to move a Variable onto the GPU from a multithreaded
    collate call.

    This is a callable class, the intended usage is:
      prep4rnn = PrepareForRNN(vocab_dict)
      inputs = prep4rnn(sents)

    Subclasses should override __call__. Main action in _prepare.
    """

    def __init__(self, vocab_dict):
        """Create a new CollateForRNN.

        Args:
          vocab_dict: Dictionary with entries for ext.vocab.[SOS, EOS, PADDING].
        """
        self.vocab_dict = vocab_dict

    def __call__(self, sents):
        """Prepare sents for LSTM processing.

        Args:
          sents: List of Lists of String tokens.

        Returns:
          sents (List of numpy.ndarrays of Integers), lens (numpy.ndarray of
            Integers), rev_ix_sort (numpy.ndarray of Integers).
        """
        return self._prepare(sents)

    def _add_sos_eos(self, seq):
        # seq: List of Integers
        # returns: List of Integers
        return [self.vocab_dict[vocab.SOS]] + seq + [self.vocab_dict[vocab.EOS]]

    def _pad(self, seq, length):
        # seq: List of Integers
        # length: Integer, the desired length
        # returns: List of Integers
        while len(seq) < length:
            seq.append(self.vocab_dict[vocab.PADDING])
        return seq

    def _prepare(self, sents):
        # sents: List of Lists of Integer vocab_ixs
        # returns: sents (list of arrays), lens (array), rev_ix_sort (array)

        # Convert tokens to vocab_ixs
        sents = [[self.vocab_dict[t] for t in s]
                 for s in sents]

        # Calculate sentence lengths
        lens = np.array([len(s) for s in sents])
        max_len = np.max(lens)

        # Add 2 to the lens since we add SOS and EOS, and arrayify
        lens = np.array([l + 2 for l in lens])

        # Sort by sentence length and determine ix_sort list
        sents, lens, ix_sort = self._sort_by_len(sents, lens)

        # Get reverse ix sort for rearranging in original order
        rev_ix_sort = np.argsort(ix_sort)

        # Pad sequences and add SOS-EOS
        sents = [self._pad(s, max_len) for s in sents]
        sents = [self._add_sos_eos(s) for s in sents]

        # Convert sents to list of numpy arrays
        sents = [np.array(s) for s in sents]

        return sents, lens, rev_ix_sort

    def _sort_by_len(self, sents, lens):
        # sents: List of List of Integers.
        # lens: 1D numpy.ndarray.
        # returns: sents (List of Lists of Integers), lens (numpy.ndarray of
        #   Integers), all sorted from longest to shorted sentence.
        lens, ix_sort = np.sort(lens)[::-1], np.argsort(-lens)
        sents = list(np.array(sents)[ix_sort])
        return sents, lens, ix_sort


class PrepareRNNBatch:
    """For preparing RNN batches INSIDE nn.Modules - not pre-collation."""

    def __init__(self, embeddings):
        """Create a new PrepareRNNBatch.

        Args:
          embeddings: torch.nn.Embedding.
        """
        self.embeddings = embeddings

    def __call__(self, sents):
        """Perform prepare function.

        Args:
          sents: List of numpy.ndarrays of Integers.

        Returns:
          Tensor of shape (max_len, batch_size, embed_size).
        """
        # Convert inputs to Variables
        sents = [Variable(torch.from_numpy(s)) for s in sents]

        # Stack the sequences into columns (timesteps from top to bottom)
        if torch.cuda.is_available():
            sents = torch.stack(sents, dim=1).cuda()
        else:
            sents = torch.stack(sents, dim=1)

        # Lookup embeddings
        sents = self.embeddings(sents)

        return sents


def get_embeddings(embedding_matrix, requires_grad):
    embeddings = nn.Embedding(
        embedding_matrix.shape[0],
        embedding_matrix.shape[1],
        sparse=False)
    embeddings.weight = nn.Parameter(torch.from_numpy(embedding_matrix),
                                     requires_grad=requires_grad)
    return embeddings


def init_param(param, gain=1.):
    if param.data.dim() > 1:
        nn.init.xavier_uniform(param, gain)
    else:
        param.data = torch.zeros(param.data.size())


def param_group(params, name, lr, lr_factor=1., weight_decay=0.):
    return {'params': params, 'name': name, 'lr': lr, 'lr_factor': lr_factor,
            'weight_decay': weight_decay}


def params_signature(model):
    sig = []
    for param in model.parameters():
        if param.data.dim() > 1:
            sig.append(param.data[0][0])
        else:
            sig.append(param.data[0])
    return sig


class SuperiorLSTMCell(nn.Module):
    pass


class Saver:
    """For loading and saving PyTorch models.

    TODO: this is the old version that does not do optim state dict...
    """

    def __init__(self):
        pass

    def load(self, module, path):
        # module could be a model or an optimizer
        print('Loading checkpoint at %s...' % path)
        module.load_state_dict(torch.load(path))

    def save(self, module, path):
        torch.save(module.state_dict(), path)
