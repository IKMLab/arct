"""For creating vocab dictionaries and word embedding matrices."""
import numpy as np


def create_embeddings(vocab_dict, emb_size, embedding_file_path):
    """Create embeddings for the vocabulary.

    Creates an embedding matrix given the pre-trained word vectors, and any OOV
    tokens are initialized to random vectors.

    Args:
      vocab_dict: Dictionary for the vocab with {token: id}.
      emb_size: Integer, the size of the word embeddings.
      embedding_file_path: String, file path to the pre-trained embeddings to
        use.

    Returns:
      embeddings, oov: 2D numpy.ndarray of shape vocab_size x emb_size,
        Dictionary of OOV vocab items.
    """
    print('Creating word embeddings from %s...' % embedding_file_path)
    vocab_size = max(vocab_dict.values()) + 1
    print('vocab_size = %s' % vocab_size)
    oov = dict(vocab_dict)
    embeddings = np.random.normal(size=(vocab_size, emb_size))\
        .astype('float32', copy=False)
    with open(embedding_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            s = line.split()
            if len(s) > 301:  # a hack I have seemed to require for GloVe 840B
                s = [s[0]] + s[-300:]
                assert len(s) == 301
            if s[0] in vocab_dict.keys():
                if s[0] in oov.keys():  # seems we get some duplicate vectors.
                    oov.pop(s[0])
                try:
                    embeddings[vocab_dict[s[0]], :] = np.asarray(s[1:])
                except Exception as e:
                    print('i: %s' % i)
                    print('s[0]: %s' % s[0])
                    print('vocab_[s[0]]: %s' % vocab_dict[s[0]])
                    print('len(vocab): %s' % len(vocab_dict))
                    print('vocab_min_val: %s' % min(vocab_dict.values()))
                    print('vocab_max_val: %s' % max(vocab_dict.values()))
                    raise e
    print('Success.')
    print('OOV count = %s' % len(oov))
    print(oov)
    return embeddings, oov
