"""Base classes for training."""
import time
import numpy as np
import torch
import os


DIV = '--------\t  ----------------\t------------------\t--------\t--------'


def pretty_time(secs):
    """Get a readable string for a quantity of time.

    Args:
      secs: Integer, seconds.

    Returns:
      String.
    """
    if secs < 60.0:
        return '%4.2f secs' % secs
    elif secs < 3600.0:
        return '%4.2f mins' % (secs / 60)
    elif secs < 86400.0:
        return '%4.2f hrs' % (secs / 60 / 60)
    else:
        return '%3.2f days' % (secs / 60 / 60 / 24)


class TrainingHistory:
    """Wraps training history details, facilitating resumption of training."""

    def __init__(self, name):
        self._id = name
        self.name = name
        self.global_step = 0
        self.global_epoch = 0
        self.epoch_losses = []
        self.epoch_accs = []
        self.tuning_accs = []
        self.t_avg_step = 0.
        self.t_avg_epoch = 0.
        self.t_total = 0.
        self.lr = 0.  # keep track of principal learning rate


class HistoryManager:
    """For loading and saving histories."""

    def __init__(self, repo):
        """Create a new HistoryManager.

        Args:
          repo: hsdbi.mongo.MongoRepository, for access to the collection that
            stores the records.
        """
        self.repo = repo

    def delete(self, name):
        self.repo.delete(name=name)

    def exists(self, name):
        return self.repo.exists(name=name)

    def load(self, name):
        json = self.repo.get(name=name)
        history = TrainingHistory(json['name'])
        for key, value in json.items():
            setattr(history, key, json[key])
        return history

    def save(self, history):
        if self.exists(history.name):
            self.repo.update(history.__dict__)
        else:
            self.repo.add(**history.__dict__)


class Trainer:
    """Base Trainer defining interface and implementing common functionality."""

    def __init__(self, model, config, train_loader, tune_loader, annealing,
                 stopping, saver, logger, history, history_manager, is_grid):
        """Create a new Trainer.

        We expect config to have a 'max_epochs'.

        Args:
          model: torch.nn.Module, the model to train.
          config: Dictionary of configuration settings.
          train_loader: torch.utils.data.dataloader.DataLoader.
          tune_loader: torch.utils.data.dataloader.DataLoader. Can be None, in
            which case the tuning part of the algorithm will not run.
          annealing: LearningRateAlgorithm.
          stopping: EarlyStoppingAlgorithm.
          saver: Saver, for checkpointing.
          logger: Logger, for logging.
          history: TrainingHistory.
          history_manager: HistoryManager, for saving the history.
          is_grid: Bool, indicates a gridsearch case - won't save params.

        Raises:
          ValueError: if config missing 'max_epochs'.
        """
        if 'max_epochs' not in config.keys():
            raise ValueError("Missing 'max_epochs' from config")

        self.max_epochs = config['max_epochs']
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.tune_loader = tune_loader
        self.annealing = annealing
        self.stopping = stopping
        self.saver = saver
        self.logger = logger
        self.history = history
        history.lr = config['lr']
        self.history_manager = history_manager
        self.batches_per_epoch = len(train_loader)
        self.is_grid = is_grid

    def epoch_ending(self, tuning_acc):
        self.logger.log(DIV)
        self.history.t_avg_step = (self.history.t_avg_step
                                   + np.average(self.step_times)) / 2
        self.history.t_avg_epoch = (self.history.t_avg_epoch
                                    + np.average(self.step_times)
                                    * len(self.train_loader)) / 2
        self.history.t_total += np.sum(self.step_times)
        self.history.epoch_losses.append(np.average(self.step_losses))
        self.history.epoch_accs.append(np.average(self.step_accs))
        self.logger.log('\t\t\t\t\t\t\t\t%s'
                        % pretty_time(self.history.t_avg_epoch))
        self.logger.log('Tuning acc: %6.4f' % tuning_acc)
        if self.annealing.update_granularity == 'epoch':
            if self.annealing.update_required(self.history):
                self.update_learning_rate()
        if not self.is_grid:
            is_best = tuning_acc == np.max(self.history.tuning_accs)
            # always save latest
            self.saver.save(self.model, is_best=False)
            if is_best:
                self.saver.save(self.model, is_best=True)
            self.history_manager.save(self.history)

    def epoch_starting(self):
        self.step_losses = []
        self.step_accs = []
        self.step_times = []
        self.history.global_epoch += 1
        self.local_step = 1
        self.model.train()
        self.logger.log(DIV)
        self.logger.log('Epoch %s \t       loss       \t     accuracy     '
                        '\tt(avg.)\t\tremaining' % self.history.global_epoch)
        self.logger.log(
            '        \t  last      avg.  \t  last      avg.  \t       \t')
        self.logger.log(DIV)

    def step_ending(self, loss, acc):
        self.step_end_time = time.time()
        time_taken = self.step_end_time - self.step_start_time
        self.step_times.append(time_taken)
        self.step_losses.append(loss)
        self.step_accs.append(acc)
        if self.history.global_epoch > 1:        
            prev_avg_time = self.history.t_avg_step
            prev_avg_loss = np.average(self.history.epoch_losses)
            prev_avg_acc = np.average(self.history.epoch_accs)
            avg_time = (prev_avg_time + np.average(self.step_times)) / 2
            avg_loss = (prev_avg_loss + np.average(self.step_losses)) / 2
            avg_acc = (prev_avg_acc + np.average(self.step_accs)) / 2
        else:
            avg_time = np.average(self.step_times)
            avg_loss = np.average(self.step_losses)
            avg_acc = np.average(self.step_accs)
        report, percent, steps_remaining = self.step_status()
        if report:
            self.logger.log('%2.0f%%:\t\t%8.4f  %8.4f\t%6.4f%%  %6.4f%%\t%s\t%s'
                            % (percent,
                               loss,
                               avg_loss,
                               acc * 100,
                               avg_acc * 100,
                               pretty_time(avg_time),
                               pretty_time(avg_time * steps_remaining)))
        self.local_step += 1
        if self.annealing.update_granularity == 'step':
            if self.annealing.update_required(self.history):
                self.update_learning_rate()

    def step_starting(self):
        self.history.global_step += 1
        self.step_start_time = time.time()

    def step_status(self):
        report_interval = np.ceil(self.batches_per_epoch / 10.)
        percent = self.local_step / report_interval * 10
        steps_remaining = self.batches_per_epoch - self.local_step
        report = self.local_step % report_interval == 0 \
            and self.local_step < report_interval * 10
        return report, percent, steps_remaining

    def step(self, batch):
        """Take a training step.

        Args:
          batch: an object representing an appropriate batch for the model.

        Returns:
          loss: Float.
          acc: Float.
        """
        self.model.zero_grad()
        _, loss, acc = self.model.forward(batch)
        self.model.optimize(loss)
        loss = float(loss.cpu().data.numpy()[0])
        acc = float(acc.cpu().data.numpy()[0])
        return loss, acc

    def stop(self):
        if self.history.global_epoch >= self.max_epochs:
            self.logger.log('Global max epoch reached (%s).' % self.max_epochs)
            return True
        early_stop, info = self.stopping(self.history)
        if early_stop:
            self.logger.log('Stopping condition met:')
            self.logger.log(info)
            return True
        return False

    def train(self):
        """Executes training algorithm.

        Returns:
          Float: best accuracy on tuning set.
        """
        if torch.cuda.is_available():
            self.model.cuda()
        while not self.stop():
            self.epoch_starting()
            for _, batch in enumerate(self.train_loader):
                self.step_starting()
                loss, acc = self.step(batch)
                self.step_ending(loss, acc)
            tuning_acc = self.tuning()
            self.epoch_ending(tuning_acc)
        return np.max(self.history.tuning_accs)

    def tuning(self):
        if self.tune_loader:
            self.model.eval()
            cum_acc = 0.
            for _, batch in enumerate(self.tune_loader):
                _, __, acc = self.model.forward(batch)
                cum_acc += acc.cpu().data.numpy()[0]
            acc = cum_acc / len(self.tune_loader)
            self.history.tuning_accs.append(acc)
            return acc

    def update_learning_rate(self):
        # principal learning rate
        current = self.history.lr
        new = self.annealing.new_learning_rate(current, self.history)
        self.history.lr = new
        self.logger.log('LR Update :: %s \t\t%1.2e => %1.2e'
                        % ('principal', current, new))
        # individual params
        for param_group in self.model.optimizer.param_groups:
            current = param_group['lr']
            new = self.annealing.new_learning_rate(current, self.history)
            param_group['lr'] = new
            self.logger.log('LR Update :: %s %s\t%1.2e => %1.2e'
                            % (param_group['name'],
                               '\t' if len(param_group['name']) < 10 else '',
                               current,
                               new))


class Saver:
    """For loading and saving state dicts.

    The load and save methods accept a model argument. This model is expected to
    be a torch.nn.Module that additionally defines "name" (String) and
    "optimizer" (PyTorch optimizer module) attributes.

    The checkpoint locations are given by the ckpt_dir argument given to the
    constructor, which defines the base directory, and then given the model.name
    attribute saves the files to:
      ckpt_dir/model.name_{model, optim}_{best, latest}
    """

    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir

    def ckpt_path(self, name, module, is_best):
        """Get the file path to a checkpoint.

        Args:
          name: String, the model name.
          module: String in {model, optim}.
          is_best: Bool.

        Returns:
          String.
        """
        return os.path.join(
            self.ckpt_dir,
            '%s_%s_%s' % (name, module, 'best' if is_best else 'latest'))

    def load(self, model, is_best=False, load_to_cpu=False):
        """Load model and optimizer state dict.

        Args:
          model: a PyTorch model that defines an optimizer attribute.
          is_best: Bool, indicates whether the checkpoint is the best tuning
            accuracy. If not it is "latest".
          load_to_cpu: Bool, for loading GPU trained parameters on cpu. Default
            is False.
        """
        model_path = self.ckpt_path(model.name, 'model', is_best)
        optim_path = self.ckpt_path(model.name, 'optim', is_best)
        if not torch.cuda.is_available() or load_to_cpu:
            model_state_dict = torch.load(
                model_path, map_location=lambda storage, loc: storage)
            optim_state_dict = torch.load(
                optim_path, map_location=lambda storage, loc: storage)
        else:
            model_state_dict = torch.load(model_path)
            optim_state_dict = torch.load(optim_path)
        model.load_state_dict(model_state_dict)
        model.optimizer.load_state_dict(optim_state_dict)

    def save(self, model, is_best):
        """Save a model and optimizer state dict.

        Args:
          model: a PyTorch model that defines an optimizer attribute.
          is_best: Bool, indicates whether the checkpoint is the best tuning
            accuracy. If not it is "latest".
        """
        model_path = self.ckpt_path(model.name, 'model', is_best)
        optim_path = self.ckpt_path(model.name, 'optim', is_best)
        torch.save(model.state_dict(), model_path)
        torch.save(model.optimizer.state_dict(), optim_path)
