"""Base model and trainer for TensorFlow, plus utility functions."""
import tensorflow as tf
import os
import functools
from ext import models, training
import math


# Utility Functions


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


def clip_gradients(grads_and_vars, norm=3.0, axes=0):
    """Clip gradients.

    Args:
      grads_and_vars: the gradients and variables from the optimizer.
      norm: Float, the value to clip at - default 3.0.
      axes: Integer, the axis along which to clip - default 0.

    Returns:
      List of clipped gradients.
    """
    return [(tf.clip_by_norm(gv[0],
                             clip_norm=norm,
                             axes=axes), gv[1])
            for gv in grads_and_vars
            if gv[0] is not None]


def layer_params(d_in, d_out, activation):
    limit = calculate_limit(d_in, d_out, activation)
    W = tf.Variable(tf.random_uniform(
        shape=[d_in, d_out],
        minval=-limit,
        maxval=limit))
    b = tf.Variable(tf.zeros([1, d_out]))
    return W, b


# DECORATORS FOR MODELS
# These decorators were designed for TensorFlow by Danijar Hafner:
# https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2


def doublewrap(function):
    """Decorator for a decorator.
    Allowing to use the decorator to be used without parentheses if not
    arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """A decorator for functions that define TensorFlow operations.
    The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


# Blocks


def adam_with_grad_clip(learning_rate, loss, parameters, grad_clip_norm):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # items in grads_and_vars: gv[0] = gradient; gv[1] = variable.
    grads_and_vars = optimizer.compute_gradients(
        loss,
        parameters)
    if grad_clip_norm > 0.0:
        grads_and_vars = clip_gradients(
            grads_and_vars=grads_and_vars,
            norm=grad_clip_norm)
    return optimizer.apply_gradients(grads_and_vars)


def cross_entropy_with_l2(labels, logits, _lambda, parameters):
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits,
            name='sparse_softmax_cross_entropy'))
    penalty_term = tf.multiply(
        tf.cast(_lambda, tf.float32),
        sum([tf.nn.l2_loss(p) for p in parameters]),
        name='penalty_term')
    return tf.add(cross_entropy, penalty_term, name='loss')


# Model


class TensorFlowModel(models.Model):
    """Base model for TensorFlow models.

    Attributes:
      in_training: Boolean indicating whether in training. Set to False by
        default. The methods eval() and train() will set this externally.
        This mirrors the pytorch Module class.
    """

    def __init__(self, config, vocab_dict, embed_mat):
        """Create a new TensorFlowModelBase.

        Args:
          config: coldnet.models.Config object with config settings.
          vocab_dict: Dictionary, {token: id}, for the corpus.
          embed_mat: 2D numpy.ndarray, embedding matrix.
        """
        super(TensorFlowModel, self).__init__(
            framework='tf',
            config=config)
        self.vocab_dict = vocab_dict
        self.vocab_size = len(vocab_dict)
        self.embeddings = tf.Variable(embed_mat, self.tune_embeddings)
        # Generate keep probability dict from config
        self.p_keep = {}
        for key in config.dropout_keys():
            name = key.split('_')[-1]
            self.p_keep[name] = tf.placeholder(tf.float32, [], name=name)
        self.in_training = False

    def biases(self):
        return [v for
                v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if v.name.endswith('biases:0')]

    def _init_backend(self):
        # Initializes all common backend parts of the computation graph
        # define on this class. It is convenient to call this at the end
        # of the constructor on a child model class.
        self.logits
        self.loss
        self.optimize
        self.predictions
        self.correct_predictions
        self.accuracy
        self.confidences

    @define_scope
    def accuracy(self):
        return tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

    @define_scope
    def confidences(self):
        return tf.reduce_max(self.logits, axis=1)

    @define_scope
    def correct_predictions(self):
        return tf.equal(self.predictions, tf.argmax(self.labels,
                                                    axis=0,
                                                    output_type=tf.int32))

    def dropout_feed_dict(self):
        dropout_keys = [key for key in self.config.keys()
                        if key.startswith('p_keep_')]
        dropout_names = [key.split('_')[-1] for key in dropout_keys]
        keys_names = dict(zip(dropout_keys, dropout_names))
        if self.in_training:
            return {self.p_keep[name]: self.config[key]
                    for key, name in keys_names.items()}
        else:
            return {self.p_keep[name]: 1.0
                    for name in dropout_names}

    def eval(self):
        self.in_training = False

    @define_scope
    def labels(self):
        # default implementation safely this.
        return tf.placeholder(
            tf.int32,
            [None],
            name='labels')

    @define_scope
    def logits(self):
        raise NotImplementedError

    @define_scope
    def loss(self):
        raise NotImplementedError

    @define_scope
    def optimize(self):
        return adam_with_grad_clip(
            learning_rate=self.learning_rate,
            loss=self.loss,
            parameters=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
            grad_clip_norm=self.grad_clip_norm)

    def parameters(self):
        return self.biases() + self.weights()

    @define_scope
    def predictions(self):
        return tf.argmax(self.logits, axis=1)

    def train(self):
        self.in_training = True

    def weights(self):
        return [v for
                v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if v.name.endswith('weights:0')]

    def word2vec(self, indices):
        vecs = tf.nn.embedding_lookup(self.embeddings, indices)
        dropped_vecs = tf.nn.dropout(vecs, self.p_keep['input'])
        return dropped_vecs


# Trainer


class TensorFlowTrainer(training.TrainerBase):
    """Base class for training TensorFlow models."""

    def __init__(self, model, history, train_data, tune_data, ckpt_dir, dbi):
        super(TensorFlowTrainer, self).__init__(
            model, history, train_data, tune_data, dbi)
        self.ckpt_dir = ckpt_dir
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _checkpoint(self, is_best):
        path = training.model_path(self.ckpt_dir, self.history.name, is_best)
        self.saver.save(
            self.sess,
            path,
            global_step=self.history.global_step)

    def _load_last(self):
        path = training.model_path(self.ckpt_dir, self.history.name, False)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('Checkpoint "%s" not found' % path)

    # NOTE: step and predict not implemented - need to be.
