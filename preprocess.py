"""Pre-processing the data for this task."""
from ext import vocab_emb as ve
from ext import pickling as pkl
from arct import data, glovar


"""
Script for performing all pre-processing steps.
You must:
1) Make sure all the vector files are in the folders specified in glovar.VEC_DIR
2) Make sure you have specified your glovar.DATA_DIR

The DATA_DIR will then be populated with pickles for all word vector types and
the vocab dict. These can be accessed in arct.data functions vocab() and
embeddings().
"""


# Create vocab dict
print('Creating vocab dict...')
vocab, counter = ve.create_vocab_dict(data.text())
pkl.save(vocab, glovar.DATA_DIR, 'vocab_dict.pkl')
pkl.save(counter, glovar.DATA_DIR, 'word_counter.pkl')
print('Success. Vocab length: %s' % len(vocab))


# Create embeddings
oov_counts = dict()
for emb_fam in ve.WORD_VECS.keys():
    oov_counts[emb_fam] = dict()
    for emb_type in ve.WORD_VECS[emb_fam].keys():
        for emb_size in ve.WORD_VECS[emb_fam][emb_type].keys():
            emb_mat, oov_count = ve.create_embeddings(
                vocab, emb_fam, emb_type, emb_size, glovar.VEC_DIR[emb_fam])
            oov_counts[emb_fam][emb_type] = oov_count
            pkl.save(emb_mat, glovar.DATA_DIR, 'emb_%s_%s_%s.pkl'
                     % (emb_fam, emb_type, emb_size))
pkl.save(oov_counts, glovar.DATA_DIR, 'oov_counts.pkl')
