"""Pre-processing the data for this task."""
from ext import vocab_emb as ve
from ext import pickling as pkl
from arct import data
from arct import glovar as gv


# Create vocab dict
print('Creating vocab dict...')
vocab, counter = ve.create_vocab_dict(data.text())
print('Success. Vocab length: %s' % len(vocab))


# Create embeddings
oov_counts = dict()
for emb_fam in ve.WORD_VECS.keys():
    oov_counts[emb_fam] = dict()
    for emb_type in ve.WORD_VECS[emb_fam].keys():
        for emb_size in ve.WORD_VECS[emb_fam][emb_type].keys():
            emb_mat, oov_count = ve.create_embeddings(
                vocab, emb_fam, emb_type, emb_size, gv.VEC_DIR[emb_type])
            oov_counts[emb_fam][emb_type] = oov_count
pkl.save(oov_counts, gv.DATA_DIR, 'oov_counts.pkl')
