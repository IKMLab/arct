"""Factories for object creation from config."""
from torch.utils.data import dataloader
from ext import training, collating, stopping, annealing, logging, pickling, \
                experiments, tokenization
import nltk
import glovar
import os


#
# Data Factories


class DataFactory:
    """For creating data related objects."""

    def __init__(self):
        pass

    def data(self, name):
        """Get a dataset.

        Args:
          name: String, an identifying name for the data - e.g. "train".

        Returns:
          List.
        """
        raise NotImplementedError

    def test(self, config):
        """Get testing data.

        Returns:
          torch.utils.data.dataset.Dataset.
        """
        raise NotImplementedError

    def train(self, config):
        """Get training data.

        Returns:
          torch.utils.data.dataset.Dataset.
        """
        raise NotImplementedError

    def tune(self, config):
        """Get tuning data.

        Returns:
          torch.utils.data.dataset.Dataset.
        """
        raise NotImplementedError


class TextDataFactory(DataFactory):
    """Creating data related objects for text datasets."""

    def __init__(self):
        super(TextDataFactory, self).__init__()

    def embeddings(self, config):
        print(config)
        return pickling.load(
            glovar.DATA_DIR, config['embed_type'], [config['target']])

    def vocab(self, config):
        return pickling.load(glovar.DATA_DIR, 'vocab', [config['target']])

    def wordset(self, tokenizer):
        raise NotImplementedError


#
# Train Factories


class TrainFactory:
    """For creating objects for training from config."""

    def __init__(self, data_factory, history_manager, experiment_manager):
        """Create a new Factory."""
        self.data_factory = data_factory
        self.history_manager = history_manager
        self.experiment_manager = experiment_manager

    def annealing(self, config):
        if config['annealing'] == 'const':
            return annealing.ConstantLearningRate(config)
        # TODO: complete all options
        elif config['annealing'] == 'tune_acc_decay':
            return annealing.TuningAccDecay(config)
        else:
            ValueError('Unexpected annealing choice "%r"' % config['annealing'])

    def collator(self, config):
        if config['collator'] == 'none':
            return None
        else:
            raise ValueError('Unexpected collator "%r"' % config['collator'])

    def data_loader(self, config, dataset, collator):
        return dataloader.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            num_workers=4,
            collate_fn=collator)

    def experiment(self, config):
        #if self.experiment_manager.exists(config['name']):
        #    return self.experiment_manager.load(config)
        return experiments.Experiment(config.__dict__)

    def history(self, config, run_num):
        name = config['name'] + str(run_num)
        if self.history_manager.exists(name):
            if config['override']:
                print('Overriding - deleting previous history...')
                self.history_manager.delete(name)
            else:
                return self.history_manager.load(name)
        return training.TrainingHistory(name)

    def logger(self, config, run_num):
        name = config['name'] + str(run_num)
        path = os.path.join(glovar.LOG_DIR, name + '.log')
        return logging.Logger(path, override=config['override'])

    def model(self, config, run_num):
        raise NotImplementedError

    def saver(self, config):
        return training.Saver(glovar.CKPT_DIR)

    def stopping(self, config):
        if config['stopping'] == 'none':
            return stopping.EarlyStopping(config)
        elif config['stopping'] == 'min_lr':
            return stopping.MinLR(config)
        else:
            raise ValueError('Unexpected stopping algorithm: %r'
                             % config['stopping'])

    def trainer(self, config, run_num, is_grid):
        """Get a Trainer.

        Args:
          config: Dictionary.
          run_num: Integer.

        Returns:
          Trainer.
        """
        model = self.model(config, run_num)
        collator = self.collator(config)
        train_data = self.data_factory.train(config)
        tune_data = self.data_factory.tune(config)
        print('Train dataset size: %s' % len(train_data))
        print('Tune dataset size: %s' % len(tune_data))
        train_loader = self.data_loader(config, train_data, collator)
        tune_loader = self.data_loader(config, tune_data, collator)
        annealing = self.annealing(config)
        stopping = self.stopping(config)
        saver = self.saver(config)
        logger = self.logger(config, run_num)
        history = self.history(config, run_num)
        if history.global_epoch > 0:
            saver.load(model, is_best=False)
        return training.Trainer(model, config, train_loader, tune_loader,
                                annealing, stopping, saver, logger, history,
                                self.history_manager, is_grid)


class TextTrainFactory(TrainFactory):
    """Extends TrainFactory specifically for text based datasets."""

    def __init__(self, data_factory, history_manager, experiment_manager):
        super(TextTrainFactory, self).__init__(
            data_factory, history_manager, experiment_manager)

    def collator(self, config):
        vocab = self.data_factory.vocab(config)
        tokenizer = self.tokenizer(config)
        if config['collator'] == 'none':
            return None
        elif config['collator'] == 'sent':
            return collating.CollateSents(
                vocab=vocab,
                tokenizer=tokenizer,
                sos_eos=config['sos_eos'])
        elif config['collator'] == 'rnn_sent':
            return collating.CollateSentsForRNN(
                vocab=vocab,
                tokenizer=tokenizer,
                sos_eos=config['sos_eos'])
        else:
            raise ValueError('Unexpected collator: "%r"' % config['collator'])

    def tokenizer(self, config):
        if config['tokenizer'] == 'nltk':
            return nltk.word_tokenize
        elif config['tokenizer'] == 'spacy':
            return tokenization.SpaCyTokenizer()
        else:
            raise ValueError('Unexpected tokenizer: "%r"' % config['tokenizer'])
