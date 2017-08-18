import math
import tensorflow as tf
from ext import tensor_flow


def calculate_gain(activation):
    if activation == 'None':
        return 1
    elif activation == 'sigmoid':
        return 1
    elif activation == 'tanh':
        return 5. / 3.
    elif activation == 'relu':
        return math.sqrt(2)
    else:
        raise ValueError('Unexpected activation "%s"' % activation)


def calculate_limit(fan_in, fan_out, activation):
    gain = calculate_gain(activation)
    n = (fan_in + fan_out) / 2.0
    return math.sqrt(3.0 * gain / n)


def layer_params(d_in, d_out, activation):
    limit = calculate_limit(d_in, d_out, activation)
    W = tf.Variable(tf.random_uniform(
        shape=[d_out, d_in],
        minval=-limit,
        maxval=limit))
    b = tf.Variable(tf.zeros([d_out, 1]))
    return W, b


class BiLSTMEncoder(tensor_flow.TensorFlowModel):
    def __init__(self, config, vocab_dict, embed_mat):
        super(BiLSTMEncoder, self).__init__(config, vocab_dict, embed_mat)

        # Parameters
        self.W1_comp, self.b1_comp = layer_params(2 * self.hidden_size,
                                                  self.hidden_size,
                                                  'relu')
        self.W2_comp, self.b2_comp = layer_params(self.hidden_size,
                                                  self.hidden_size,
                                                  'relu')
        self.W_eval, self.b_eval = layer_params(self.hidden_size,
                                                self.hidden_size,
                                                'None')

        def compose(vecs):
            h_in = tf.concat(vecs, axis=1)
            h1 = tf.nn.relu(h_in.mm(self.W1_comp) + self.b1_comp)
            h2 = tf.nn.relu(h1.mm(self.W2_comp) + self.b2_comp)
            return h2

        # Inputs
        # They should be processable in parallel.
        # So: [current_batch_size, 4, current_seq_length]
        self.inputs = tf.placeholder(tf.float32, [None, 4, None], 'inputs')
        # Otherwise:
        self.reasons = tf.placeholder(tf.float32, [None, 1, None], 'reasons')
        self.claims = tf.placeholder(tf.float32, [None, 1, None], 'claims')
        self.warrants = tf.placeholder(tf.float32, [None, 1, None], 'warrants')
        self.alt_warrants = tf.placeholder(
            tf.float32, [None, 1, None], 'alt_warrants')

        # Word2vec
        self.R_vecs = self.word2vec(self.reasons)
        self.C_vecs = self.word2vec(self.claims)
        self.W_vecs = self.word2vec(self.warrants)
        self.AW_vecs = self.word2vec(self.alt_warrants)

        # Encoding
        self.R = biLSTM(
            self.R_vecs, self.hidden_size, self.R_vecs.shape[1], 'R')
        self.C = biLSTM(
            self.C_vecs, self.hidden_size, self.C_vecs.shape[1], 'C')
        self.W = biLSTM(
            self.W_vecs, self.hidden_size, self.W_vecs.shape[1], 'W')
        self.AW = biLSTM(
            self.AW_vecs, self.hidden_size, self.AW_vecs.shape[1], 'AW')

        # Compose Arguments
        self.RW = compose([self.R, self.W])
        self.RAW = compose([self.R, self.AW])
        self.CRW = compose([self.RW, self.C])
        self.CRAW = compose([self.RAW, self.C])



        self._init_backend()


def biLSTM(inputs, dim, seq_len, name):
    """
    A Bi-Directional LSTM layer. Returns forward and backward hidden states as a tuple, and cell states as a tuple.
    Ouput of hidden states: [(batch_size, max_seq_length, hidden_dim), (batch_size, max_seq_length, hidden_dim)]
    Same shape for cell states.
    """
    with tf.name_scope(name):
        with tf.variable_scope('forward' + name):
            lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=dim)
        with tf.variable_scope('backward' + name):
            lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=dim)

        hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fwd, cell_bw=lstm_bwd, inputs=inputs, sequence_length=seq_len, dtype=tf.float32, scope=name)
        h_bi = tf.concat(hidden_states, axis=2)
        h_sum = tf.reduce_sum(h_bi, 1)
        h_ave = tf.div(h_sum, tf.expand_dims(
            tf.cast(seq_len, tf.float32), -1))

    return h_ave



