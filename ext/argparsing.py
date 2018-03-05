"""Utility for command line arg parsing with a config dict."""
import argparse


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/
    parsing-boolean-values-with-argparse
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def arg_type(x):
    if type(x) == bool:
        return str2bool
    else:
        return type(x)


def parse(config):
    """Parse command line arguments.

    Two command line arguments are always expected:
      name: String, the name of the training run / experiment.
      model: String, the type of model to use.
    The config should contain these. The remaining args in config are optional.
    All defaults arguments are given by the config passed.

    Args:
      config: Dictionary of expected configuration values.

    Returns:
      Dictionary: configuration collated with command line args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('name',
                        type=str,
                        help='The name of the training run / experiment.')
    parser.add_argument('model',
                        type=str,
                        help='The type of model.')
    for key in [k for k in config.keys() if k not in ['name', 'model']]:
        parser.add_argument('-%s' % key, '--%s' % key,
                            help='Set config.%s' % key,
                            type=arg_type(config[key]),
                            default=config[key])
    args = parser.parse_args()
    for key in config.keys():
        config[key] = getattr(args, key)
    return config
