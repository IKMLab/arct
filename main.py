from arct import preprocess


v = preprocess.vocab()
e = preprocess.create_embeddings(v, 'glove_twt', 25)
print(e.shape)

oov = preprocess.oov_counts()
print(oov)
