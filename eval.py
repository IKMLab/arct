import tensorflow as tf
import numpy as np


def length(sequence):
    """
    Courtesy of Danijar Hafner:
    https://danijar.com/variable-sequence-lengths-in-tensorflow/
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=1))
    l = tf.reduce_sum(used, reduction_indices=1)
    l = tf.cast(l, tf.int32)
    return l


_r = np.array([[1, 0], [2, 1]])
_c = np.array([[2, 1], [1, 0]])
_w = np.array([[1, 1], [2, 2]])
_aw = np.array([[2, 1], [1, 2]])
emb_mat = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]], dtype='float32')

emb = tf.Variable(emb_mat, True)
r = tf.placeholder(tf.int32, [None, None], 'r')
c = tf.placeholder(tf.int32, [None, None], 'c')
w = tf.placeholder(tf.int32, [None, None], 'w')
aw = tf.placeholder(tf.int32, [None, None], 'aw')

fd = {r: _r, c: _c, w: _w, aw: _aw}

# don't need abs, since all indices are positive
# getting the sign gives a list of ones for all non-zero, non-paddings
# then just need to find a way to sum over the right axis?

s = tf.sign(c)
l = tf.reduce_sum(s, axis=1)

a = np.array([0.3, 0.5, 0.9])
b = tf.placeholder(tf.float32, [3,])
c = tf.cast(b > 0.5, tf.int32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x = sess.run(c, {b: a})
    print(x)


raise Exception


l_r = length(r)
l_c = length(c)
l_w = length(w)
l_aw = length(aw)

_R = tf.nn.embedding_lookup(emb, r)  # [batch_size, sequence_length, embed_size]
_C = tf.nn.embedding_lookup(emb, c)
_W = tf.nn.embedding_lookup(emb, w)
_AW = tf.nn.embedding_lookup(emb, aw)

with tf.name_scope('bilstm'):
    with tf.variable_scope('forward_bilstm', reuse=True):
        lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=3)
    with tf.variable_scope('backward_bilstm', reuse=True):
        lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=3)


W_proj = tf.Variable(tf.random_uniform([6, 3]), dtype=tf.float32)
b_proj = tf.Variable(tf.zeros([1, 3]), dtype=tf.float32)
def bilstm(inputs, seq_lens, name):
    hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=lstm_fwd, cell_bw=lstm_bwd, inputs=inputs,
        sequence_length=seq_lens, dtype=tf.float32, scope='bilstm_' + name)
    h_bi = tf.concat(hidden_states, axis=2)
    h_sum = tf.reduce_sum(h_bi, 1)
    h_ave = tf.div(h_sum, tf.expand_dims(
        tf.cast(seq_lens, tf.float32), -1))
    enc = tf.matmul(h_ave, W_proj) + b_proj
    return enc


R = bilstm(_R, [1, 2], 'R')
C = bilstm(_C, [1, 2], 'C')
W = bilstm(_W, [2, 2], 'W')
AW = bilstm(_AW, [2, 2], 'AW')

W_comp = tf.Variable(tf.random_uniform([6, 3], minval=1., maxval=1.), dtype=tf.float32)
b_comp = tf.Variable(tf.zeros([1, 3]), dtype=tf.float32)


def compose(vecs):
    concat = tf.concat(vecs, axis=1)
    h = tf.nn.relu(tf.matmul(concat, W_comp) + b_comp)
    return h


RW = compose([R, W])
RAW = compose([R, AW])
RWC = compose([RW, C])
RAWC = compose([RAW, C])


args = tf.concat([RWC, RAWC], axis=1)


W_eval = tf.Variable(tf.random_uniform([6, 1]), dtype=tf.float32)
b_eval = tf.Variable(tf.zeros([1, 1]), dtype=tf.float32)

logits = tf.matmul(args, W_eval) + b_eval


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x, y = sess.run([l_r, l_w], fd)
    print('l_r')
    print(x)
    print(x.shape)
    print(x.dtype)
    print('l_w')
    print(y)
    print(y.shape)
    print(y.dtype)
