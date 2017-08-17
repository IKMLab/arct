import math
import tensorflow as tf


def calculate_gain(activation):
    if activation is None:
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


class Model:
    def __init__(self, hidden_size, embeddings, emb_train):



        # Parameters
        self.E = tf.Variable(embeddings, trainable=emb_train)

        self.W_mlp1, b_mlp1 = layer_params()
