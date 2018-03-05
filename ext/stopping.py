"""Early stopping algorithms."""


class EarlyStopping:
    """Base early stopping algorithm declaring interface.

    Default implementation simply returns False, None - i.e. no early stopping.
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, history):
        return self.stop(history)

    def stop(self, history):
        return False, None


class MinLR(EarlyStopping):
    """Early stopping when learning rate crosses a minimum threshold."""

    def __init__(self, config):
        super(MinLR, self).__init__(config)
        self.limit = config['stop_lr_lim']

    def stop(self, history):
        evaluation = history.lr < self.limit
        info = '%s < %s' % (history.lr, self.limit)
        return evaluation, info
