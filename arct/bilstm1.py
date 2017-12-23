import tensorflow as tf
from ext import tensor_flow


class BiLSTM1(tensor_flow.TensorFlowModel):
    """BiLSTM encode-compose model with single hidden layer classifier."""
    def __init__(self, config, vocab_dict, embed_mat):
        super(BiLSTM1, self).__init__(config, vocab_dict, embed_mat)

        # Inputs: size [batch_size, seq_length], both can vary, so None here.
        self.r = tf.placeholder(tf.int32, [None, None], 'reasons')
        self.c = tf.placeholder(tf.int32, [None, None], 'claims')
        self.w = tf.placeholder(tf.int32, [None, None], 'warrants')
        self.aw = tf.placeholder(tf.int32, [None, None], 'alt_warrants')

        # Parameters
        with tf.name_scope('bilstm'):
            with tf.variable_scope('forward_bilstm', reuse=True):
                self.lstm_fwd = tf.contrib.rnn.LSTMCell(
                    num_units=self.hidden_size)
            with tf.variable_scope('backward_bilstm', reuse=True):
                self.lstm_bwd = tf.contrib.rnn.LSTMCell(
                    num_units=self.hidden_size)
        with tf.variable_scope('compose'):
            # 8 x hidden_size because (4 components) x (biLSTM returns 2x).
            self.W_comp, self.b_comp = tensor_flow.layer_params(
                8 * self.hidden_size, self.hidden_size, 'relu')
        with tf.name_scope('evaluate'):
            self.W_eval, self.b_eval = tensor_flow.layer_params(
                self.hidden_size, 1, 'sigmoid')

        # Initialize computation graph.
        self._init_backend()

    def bilstm(self, inputs, seq_lens, name):
        hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.lstm_fwd, cell_bw=self.lstm_bwd, inputs=inputs,
            sequence_length=seq_lens, dtype=tf.float32, scope='bilstm_' + name)
        h_bi = tf.concat(hidden_states, axis=2)
        h_sum = tf.reduce_sum(h_bi, 1)
        h_ave = tf.div(h_sum, tf.expand_dims(
            tf.cast(seq_lens, tf.float32), -1))
        return h_ave

    @tensor_flow.define_scope
    def labels(self):
        return tf.placeholder(tf.int32, [1, None], name='labels')

    @tensor_flow.define_scope
    def logits(self):
        # seq lens
        l_r = self.seq_lens(self.r)
        l_c = self.seq_lens(self.c)
        l_w = self.seq_lens(self.w)
        l_aw = self.seq_lens(self.aw)

        # word2vec
        _R = self.word2vec(self.r)
        _C = self.word2vec(self.c)
        _W = self.word2vec(self.w)
        _AW = self.word2vec(self.aw)

        # encoding
        R = self.bilstm(_R, l_r, 'reasons_encoding')
        C = self.bilstm(_C, l_c, 'claims_encoding')
        W = self.bilstm(_W, l_w, 'warrants_encoding')
        AW = self.bilstm(_AW, l_aw, 'alt_warrants_encoding')

        # composition
        args_in = tf.concat([R, C, W, AW], axis=1)
        args_in = tf.nn.dropout(args_in, self.p_keep['input'])

        # learn intermediate representation
        h1 = tf.nn.relu(tf.matmul(args_in, self.W_comp) + self.b_comp)
        h1 = tf.nn.dropout(h1, self.p_keep['fc'])

        # classify
        logits = tf.sigmoid(tf.matmul(h1, self.W_eval) + self.b_eval)

        return logits

    @tensor_flow.define_scope
    def loss(self):
        return tf.losses.log_loss(self.labels, self.logits)

    @tensor_flow.define_scope
    def predictions(self):
        return tf.cast(self.logits > 0.5, tf.int32)

    def seq_lens(self, seqs):
        signs = tf.sign(seqs)
        sums = tf.reduce_sum(signs, axis=1)
        return sums