"""For creating vocab dictionaries and word embedding matrices."""
import numpy as np
import spacy
import collections


WORD_VECS = {
    'glove': {
        'twt': {
            25: 'glove.twitter.27B.25d.txt',
            50: 'glove.twitter.27B.50d.txt',
            100: 'glove.twitter.27B.100d.txt',
            200: 'glove.twitter.27B.200d.txt'},
        'wpg': {
            50: 'glove.6B.50d.txt',
            100: 'glove.6B.100d.txt',
            200: 'glove.6B.200d.txt',
            300: 'glove.6B.300d.txt'},
        '42B': {
            300: 'glove.42B.300d.txt'},
        '840B': {
            300: 'glove.840B.300d.txt'}
    },
    'fast': {
        'wiki': {
            300: ''},
        'sub': {
            300: ''},
        'crawl': {
            300: ''}
    }}
PADDING = "<PAD>"
UNKNOWN = "<UNK>"
LBR = '('
RBR = ')'


def create_embeddings(vocab, emb_fam, emb_type, emb_size, emb_dir):
    """Create embeddings for the vocabulary.

    Creates an embedding matrix given the pre-trained word vectors, and any OOV
    tokens are initialized to random vectors.

    Args:
      vocab: Dictionary for the vocab with {token: id}.
      emb_fam: String, embedding family, i.e. glove or fast.
      emb_type: String, a valid embedding type (see WORD_VECS above).
      emb_size: Integer, a valid embedding size given the type (see WORD_VECS).
      emb_dir: String, the directory where the embeddings are saved.

    Returns:
      2D numpy.ndarray of shape vocab_size x emb_size, Integer number of vocab
        items found to be OOV.
    """
    print('Creating %s %s word embeddings of size %s...'
          % (emb_fam, emb_type, emb_size))
    vocab_size = max(vocab.values()) + 1
    print('vocab_size = %s' % vocab_size)
    oov_count = vocab_size  # subtract as we find them
    embeddings = np.random.normal(size=(vocab_size, emb_size))
    with open(emb_dir + WORD_VECS[emb_fam][emb_type][emb_size],
              'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            s = line.split()
            if len(s) > 301:  # a hack I have seemed to require for GloVe 840B
                s = [s[0]] + s[-300:]
                assert len(s) == 301
            if s[0] in vocab.keys():
                oov_count -= 1
                try:
                    embeddings[vocab[s[0]], :] = np.asarray(s[1:])
                except Exception as e:
                    print(vocab[s[0]])
                    print(len(vocab))
                    print(min(vocab.values()))
                    print(max(vocab.values()))
                    raise Exception('%s, %s:\n%s' % (i, s[0], repr(e)))
    print('Success.')
    print('OOV count = %s' % oov_count)
    return embeddings, oov_count


def create_vocab_dict(text):
    """Create vocab dictionary.

    Args:
      text: String. Join all the text in the corpus on a space. It will be
        tokenized by SpaCy.

    Returns:
      Dictionary {token: id}, collections.Counter() with token counts.
    """
    nlp = spacy.load('en')
    doc = nlp(text)
    counter = collections.Counter()
    counter.update([t.text for t in doc])
    tokens = set([t for t in counter] + [PADDING, UNKNOWN, LBR, RBR])
    vocab_dict = dict(zip(tokens, range(len(tokens))))
    return vocab_dict, counter
