"""Learning rate annealing."""
import numpy as np


class LearningRateAlgorithm:
    """Base class for a learning rate annealing algorithm.

    To make this interface generic for consumers, all necessary hyperparameters
    are to be wrapped into the config dictionary passed to the constructor.

    The config must define a 'lr' key with the initial learning rate. This is
    pretty much relevant to all algorithms as far as I can see.

    Then during training, all information relevant to the algorithm will be
    passed via the history object.

    Attributes:
      update_granularity: String in {step, epoch}, defining when to call the
        algorithm.
      config: Dictionary of configuration settings that will contain any
        hyperparameters necessary for derived learning rate algorithms.
      initial: Float, the initial learning rate. We copy that over from the
        config for convenience.
    """

    def __init__(self, update_granularity, config):
        """Create a new LearningRateAlgorithm.

        Args:
          update_granularity: String in {step, epoch, none}, defining when to
            call the algorithm.
          config: Dictionary of configuration settings.

        Raises:
          ValueError: if update_granularity not in expected set.
          ValueError: if 'lr' not a key in config.
        """
        if update_granularity not in ['step', 'epoch', 'none']:
            raise ValueError(
                'Unexpected update granularity: %r' % update_granularity)
        if 'lr' not in config.keys():
            raise ValueError("Missing 'lr' from config.keys()")
        self.update_granularity = update_granularity
        self.config = config
        self.initial = config['lr']

    def new_learning_rate(self, current, history):
        # This function to be the one for __call__ - override this.
        raise NotImplementedError

    def update_required(self, history):
        raise NotImplementedError


class ConstantLearningRate(LearningRateAlgorithm):
    """Constant, unchanging learning rate."""

    def __init__(self, config):
        """Create a new ConstantLearningRate.

        We sent 'none' as granularity indicating the algorithm is never called
        and therefore the learning rate always remains the same.

        Args:
          config: Dictionary of configuration settings.
        """
        super(ConstantLearningRate, self).__init__(
            update_granularity='none', config=config)

    def new_learning_rate(self, current, history):
        return current

    def update_required(self, history):
        return False


class StepDecayLearningRate(LearningRateAlgorithm):
    """Learning rate decaying as a constant factor of the global step.

    http://cs231n.github.io/neural-networks-3/#anneal
    """

    def __init__(self, config):
        """Create a new StepDecayLearningRate.

        Args:
          config: Dictionary of configuration settings. We expect the keys:
            * lr_reduce_every: Integer.
            * lr_reduce_factor: Float.

        Raises:
          ValueError: if config does not contain 'lr_decay_every'.
          ValueError: if config does not contain 'lr_decay_rate'.
        """
        if 'lr_decay_every' not in config.keys():
            raise ValueError("Missing 'lr_decay_every' from config")
        if 'lr_decay_rate' not in config.keys():
            raise ValueError("Missing 'lr_decay_rate' from config")
        self.reduce_every = config['lr_decay_every']
        self.factor = config['lr_decay_rate']
        super(StepDecayLearningRate, self).__init__(
            update_granularity='step', config=config)

    def new_learning_rate(self, current, history):
        return current * self.factor

    def update_required(self, history):
        return history.global_step % self.reduce_every == 0


class EpochDecayLearningRate(LearningRateAlgorithm):
    """Learning rate decaying as a constant factor of the global step.

    http://cs231n.github.io/neural-networks-3/#anneal
    """

    def __init__(self, config):
        """Create a new EpochDecayLearningRate.

        Args:
          config: Dictionary of configuration settings. We expect the keys:
            * lr_reduce_every: Integer.
            * lr_reduce_factor: Float.

        Raises:
          ValueError: if config does not contain 'lr_decay_every'.
          ValueError: if config does not contain 'lr_decay_rate'.
        """
        if 'lr_decay_every' not in config.keys():
            raise ValueError("Missing 'lr_decay_every' from config")
        if 'lr_decay_rate' not in config.keys():
            raise ValueError("Missing 'lr_decay_rate' from config")
        self.reduce_every = config['lr_decay_every']
        self.factor = config['lr_decay_rate']
        super(EpochDecayLearningRate, self).__init__(
            update_granularity='epoch', config=config)

    def new_learning_rate(self, current, history):
        return current * self.factor

    def update_required(self, history):
        return history.global_epoch % self.reduce_every == 0


class ExponentialDecayLearningRate(LearningRateAlgorithm):
    """Decaying exponentially by step according to hyperparameters.

    http://cs231n.github.io/neural-networks-3/#anneal
    """

    def __init__(self, config):
        """Create a new ExponentialDecayLearningRate.

        Args:
          config: Dictionary of configuration settings. We expect the keys:
            * lr_decay_rate: Float.

        Raises:
          ValueError: if config does not contain 'lr_decay_rate'.
        """
        if 'lr_decay_rate' not in config.keys():
            raise ValueError("Missing 'lr_decay_rate' from config")
        self.k = config['lr_decay_rate']
        super(ExponentialDecayLearningRate, self).__init__(
            update_granularity='step', config=config)

    def new_learning_rate(self, history, current):
        return self.initial * np.exp(-self.k * history.global_step)

    def update_required(self, history):
        return True  # always required


class IterationDecay(LearningRateAlgorithm):
    """Decay each step by 1/t.

    http://cs231n.github.io/neural-networks-3/#anneal"""

    def __init__(self, config):
        """Create a new IterationDecay.

        Args:
          config: Dictionary of configuration settings. We expect the keys:
            * lr_decay_rate: Float.

        Raises:
          ValueError: if config does not contain 'lr_decay_rate'.
        """
        if 'lr_decay_rate' not in config.keys():
            raise ValueError("Missing 'lr_decay_rate' from config")
        self.k = config['lr_decay_rate']
        super(IterationDecay, self).__init__(
            update_granularity='step', config=config)

    def new_learning_rate(self, current, history):
        return self.initial / (1 + self.k * history.global_step)

    def update_required(self, history):
        return True  # always required


class TuningAccDecay(LearningRateAlgorithm):
    """Reduce learning rate if tuning accuracy decreases."""

    def __init__(self, config):
        """Create a new TuningAccDecay.

        Args:
          config: Dictionary of configuration settings. We expect the keys:
            * lr_decay_rate: Float. Will multiply, not divide.
            * lr_decay_grace: Integer, how many epochs of grace we give the
              training process before we start to enforce annealing.

        Raises:
          ValueError: if config does not contain 'lr_decay_rate'.
          ValueError: if config does not contain 'lr_decay_grace'.
        """
        if 'lr_decay_rate' not in config.keys():
            raise ValueError("Missing 'lr_decay_rate' from config")
        if 'lr_decay_grace' not in config.keys():
            raise ValueError("Missing 'lr_decay_grace' from config")
        self.k = config['lr_decay_rate']
        self.grace = config['lr_decay_grace']
        super(TuningAccDecay, self).__init__(
            update_granularity='epoch', config=config)

    def new_learning_rate(self, current, history):
        return current * self.k

    def update_required(self, history):
        if history.global_epoch >= max(self.grace, 2):
            return history.tuning_accs[-1] <= history.tuning_accs[-2]
        return False
