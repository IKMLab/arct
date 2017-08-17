"""For tracking and saving training histories."""
import numpy as np
from coldnet import models


# UTILITY FUNCTIONS


def load(dbi, name):
    """Load a History object from a MongoDb connection.

    Expects histories to be stored in a MongoDb in a db named history and a
    collection named train. The MongoDb is accessed via hsdbi.mongo.MongoFacade.

    Args:
      dbi: hsdbi.mongo.MongoFacade object, for accessing the mongo database.
      name: String, unique name of the training run.

    Returns:
      coldnet.histories.History object.
    """
    json = dbi.history.train.get(_id=name)
    config = models.Config(**json['config'])
    json.pop('config')
    history = History(name, config)
    history.name = json['_id']
    json.pop('_id')
    for key, value in json:
        setattr(history, key, value)
    return history


def save(dbi, history):
    """Save a History object to Mongo.

    Args:
      dbi: hsdbi.mongo.MongoFacade object, for accessing the mongo database.
      history: coldnet.histories.History object.
    """
    dbi.history.train.update(history.to_json())


def last_change(series):
    """Get the change between the last two elements of a series.
    Args:
      series: a List of floats usually.
    Returns:
      Float.
    """
    if len(series) == 0:
        raise ValueError('Series has no elements.')
    elif len(series) > 1:
        return series[-1] - series[-2]
    else:
        return series[0]


# HISTORY CLASS


class History:
    """Wraps config, training run name, and all training history values."""

    def __init__(self, name, config=None):
        """Create a new History.

        Args:
          name: String, unique identifying name of the training run.
          config: coldnet.models.Config. If creating a new History object,
            this cannot be None.

        Raises:
          ValueError: if name is not found and config is None.
        """
        if not config:
            raise ValueError('config cannot be None for new Histories.')
        # Global Variables
        self.name = name  # This ends up being the _id
        self.config = config
        # Epoch Variables
        self.global_epoch = 1
        self.epoch_losses = []
        self.epoch_accs = []
        self.epoch_times = []
        self.cum_epoch_loss = 0.
        self.cum_epoch_acc = 0.
        self.best_epoch_acc = 0.
        # Step Variables
        self.global_step = 1
        self.epoch_step_times = []  # only keep for one epoch
        self.cum_loss = 0.
        self.cum_acc = 0.
        # Tuning Variables
        self.tuning_accs = []

    def end_epoch(self, time_taken):
        self.epoch_times.append(time_taken)
        avg_time = np.average(self.epoch_times)
        self.epoch_losses.append(self.cum_epoch_loss)
        avg_loss = np.average(self.epoch_losses)
        change_loss = last_change(self.epoch_losses)
        self.epoch_accs.append(self.cum_epoch_acc)
        avg_acc = np.average(self.epoch_accs)
        change_acc = last_change(self.epoch_accs)
        is_best = avg_acc > self.best_epoch_acc
        if is_best:
            self.best_epoch_acc = avg_acc
        self.epoch_step_times = []
        self.cum_epoch_loss = 0.
        self.cum_epoch_acc = 0.
        self.global_epoch += 1
        return avg_time, avg_loss, change_loss, avg_acc, change_acc, is_best

    def end_step(self, time_taken, loss, accuracy):
        self.epoch_step_times.append(time_taken)
        avg_time = np.average(self.epoch_step_times)
        self.cum_loss += loss
        avg_loss = self.cum_loss / self.global_step
        self.cum_acc += accuracy
        avg_acc = self.cum_acc / self.global_step
        self.cum_epoch_loss += loss
        self.cum_epoch_acc += accuracy
        self.global_step += 1
        return self.global_step, avg_time, avg_loss, avg_acc

    def end_tuning(self, accuracy):
        self.tuning_accs.append(accuracy)
        avg_acc = np.average(self.tuning_accs)
        change_acc = last_change(self.tuning_accs)
        return avg_acc, change_acc

    def to_json(self):
        json = dict(self.__dict__)
        json.pop('name')
        json['_id'] = self.name
        json['config'] = self.config.to_json()
        return json
