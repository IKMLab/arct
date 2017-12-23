"""For building vocabs from data."""
import collections
import nltk


SOS = '<S>'
EOS = '</S>'
PADDING = "<PAD>"
UNKNOWN = "<UNK>"
LBR = '('
RBR = ')'


def create_vocab_dict(tokens):
    """Create vocab dictionary.

    Args:
      tokens: List of string tokens. The data should be pre-tokenized and
        processed such that it is ready to have every token counted and added
        to the dictionary.

    Returns:
      Dictionary {token: id}, collections.Counter() with token counts.
    """
    counter = collections.Counter()
    counter.update(tokens)
    tokens = set([t for t in counter] + [SOS, EOS, UNKNOWN, LBR, RBR])
    # Make sure 0 is padding.
    vocab_dict = dict(zip(tokens, range(1, len(tokens) + 1)))
    assert PADDING not in vocab_dict.keys()
    assert 0 not in vocab_dict.values()
    vocab_dict[PADDING] = 0
    return vocab_dict, counter


def tokenize(text):
    """Extract a token set from text.

    Tokenization is performed with .

    Args:
      text: String. A contiguous string representation of the collection.

    Returns:
      set of tokens.
    """
    return nltk.word_tokenize(text)
