"""Utilities for RNNs."""
import numpy as np


#
# Data Structures


class RNNSents:
    """Data structure for a batch of sentences.

    This structure is forced by the Pytorch RNN conventions. We need to order
    sentences by length. We therefore need a way to resort them into their
    original order to align them with their labels.
    """

    def __init__(self, sents, lens, rev_ix_sort):
        """Create a new RNNSents.

        Args:
          sents: numpy.ndarray, a 2D matrix of token ids, where each sentence is
            a row vector.
          lens: numpy.ndarray, a 1D vector of sentence lengths, ordered from
            longest to shorted
          rev_ix_sort: numpy.ndarray, a 1D vector of indices for re-sorting the
            sentences into their original order, aligned with their labels.
        """
        self.sents = sents
        self.lens = lens
        self.rev_ix_sort = rev_ix_sort


#
# Collating Utilities


class CollateSents:
    """Callable for collating sents for processing.

    Pads out sentences and optionally appends start and end of sequence.

    Args:
      sents: List of strings.

    Returns:
      np.ndarray, a 2D matrix of token ids.
    """

    def __init__(self, vocab, tokenizer, sos_eos=True):
        """Create a new CollateSents.

        Args:
          vocab: Object.
          tokenizer: Callable.
          sos_eos: Boolean, indicating whether to add start and end of sequence
            tokens.
        """
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.sos_eos = sos_eos

    def __call__(self, sents):
        sents = [self.tokenizer(sent) for sent in sents]
        sents = self.tokens_to_ixs(sents)
        lens = [len(sent) for sent in sents]
        lens = np.array(lens)
        self.pad(sents, max(lens))
        return sents

    def pad(self, sents, max_len):
        for sent in sents:
            while len(sent) < max_len:
                sent.append(self.vocab.pad_ix)

    def tokens_to_ixs(self, sents):
        if self.sos_eos:
            return [[self.vocab.sos_ix]
                    + [self.vocab[tok] for tok in sent]
                    + [self.vocab.eos_ix]
                    for sent in sents]
        else:
            return [[self.vocab[tok] for tok in sent] for sent in sents]


class CollateSentsForRNN(CollateSents):
    """Callable RNN collate function.

    This is designed to be the collate_fn argument in a DataLoader. It takes
    sentences in string format, performs tokenization, vocab lookup, sorts the
    sentences by length, and determines the reverse sort index vector. It
    returns an RNNSents data structure.

    In order to do this, we also need to pad the shorter sentences with a
    padding token. Additionally we define a flag for optional prepending and
    appending of start and end of sequence tokens. The nature of these tokens
    should be defined on the vocab object passed to the constructor.

    Args:
      sents: List of strings.

    Returns:
      RNNSents object.
    """

    def __init__(self, vocab, tokenizer, sos_eos=True):
        """Create a new CollateSentsForRNN.

        Args:
          vocab: Object.
          tokenizer: Callable.
          sos_eos: Boolean, indicating whether to add start and end of sequence
            tokens.
        """
        super(CollateSentsForRNN, self).__init__(vocab, tokenizer, sos_eos)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.sos_eos = sos_eos

    def __call__(self, sents):
        sents = [self.tokenizer(sent) for sent in sents]
        sents = self.tokens_to_ixs(sents)
        lens = [len(sent) for sent in sents]
        lens = np.array(lens)
        self.pad(sents, max(lens))
        sents, lens, ix_sort = self.sort_by_len(sents, lens)
        sents = np.stack([np.array(s) for s in sents])
        rev_ix_sort = np.argsort(ix_sort)
        return RNNSents(sents, lens, rev_ix_sort)

    def sort_by_len(self, sents, lens):
        lens, ix_sort = np.sort(lens)[::-1], np.argsort(-lens)
        sents = list(np.array(sents)[ix_sort])
        return sents, lens, ix_sort
