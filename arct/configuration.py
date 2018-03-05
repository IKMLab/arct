"""Base configuration dictionary."""


def base_config():
    """Get a basic configuration dictionary with expected keys and defaults.

    Returns:
      Dictionary.
    """
    return {
        # global config
        'name': '',              # name of the training run / experiment
        'model': '',             # type of model to load
        'annealing': 'tune_acc_decay',   # learning rate annealing algorithm
        'stopping': 'min_lr',    # type of early stopping algorithm
        'override': False,       # overrides any existing run with the same name
        'seed': -1,              # control randomization; default -1 = undefined
        'n_runs': 1,             # number of runs through training algorithm
        'collator': 'rnn_sent',  # collate function for DataLoader
        'target': 'train-full',  # the dataset name to target for training
        'tune_target': 'dev-full',  # the tune subset for training
        'from_grid': False,      # whether or not to load best grid config
        'grid_name': '',         # grid name for grid loading
        'from_grid_trans': True,  # hack for now to allow overriding this
        'train_subsample': 0,    # for selecting a smaller amount of train data
        # data
        'batch_size': 16,        # batch size
        # training
        'max_epochs': 200,       # max number of epochs for training
        'lr': 0.002,             # learning rate
        # annealing algorithms
        'lr_decay_rate': 0.2,    # rate of decay
        'lr_decay_grace': 0,     # how many epochs before decaying
        'lr_decay_every': 0.,    # number of steps between decay
        # early stopping algorithms
        'stop_lr_lim': 1e-5,     # lower limit of learning rate before stopping
        'stop_t_worse': 1,       # limit of epochs of decreasing tuning acc
        # regularization
        'l2': 0.,                # L2 regularization parameter
        'p_drop': 0.1,           # coarse-grained global dropout probability
        # model
        'hidden_size': 512,      # general hidden size setting
        'projection_size': 200,  # size of projection (for word embeddings)
        # sequence models
        'sos_eos': True,         # whether to add eos & sos tokens to sequences
        # text models
        'tokenizer': 'spacy',    # the word tokenization algorithm to use
        'embed_type': 'glove',   # embedding type: glove, fasttext, word2vec...
        'embed_size': 300,       # the size of word embeddings
        'tune_embeds': True,     # flag to tune embeddings
        'emb_lr_factor': 0.01,   # scaling factor for embedding learning rate
        # rnn encoders
        'encoder': 'lstm',       # the encoder type to use
        'encoder_layers': 1,     # number of rnn layers
        'bidirectional': True,   # whether or not the encoder is bidirectional
        'encoder_size': 512,     # encoder size (for RNNs)
        'p_drop_rnn': 0.,        # prob with which to drop h between timesteps
        'enc_lr_factor': 1.,     # scaling factor for encoder learning rate
        'tune_encoder': True,    # whether or not to tune the encoder
        # transfer
        'transfer': True,        # whether or not to transfer the encoder
        'transfer_name': 'e512',  # name for the transfer model
    }


class Config:
    """Extensible wrapper for configuration settings."""

    def __init__(self, base_config=base_config()):
        """Create a new Config."""
        for key, attr in base_config.items():
            setattr(self, key, attr)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        info = 'Config:'
        for key in sorted(self.__dict__.keys()):
            info += '\n\t%s\t\t%s%s' \
                    % (key, '\t' if len(key) < 8 else '', getattr(self, key))
        return info

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def copy(self):
        new_config = Config()
        for key, val in self.__dict__.items():
            new_config[key] = val
        return new_config

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()


def adapt(json):
    return Config(json)
