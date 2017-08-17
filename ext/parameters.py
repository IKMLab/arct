"""Parameter settings and mappings."""
import argparse


def parse_arguments(model_types, base_config):
    """Parse command line arguments.

    Any extra command line arguments (at this stage) must go into the config
    object.

    Args:
      model_types: List of valid model_type names.
      base_config: An object inheriting from coldnet.models.Config. This will
        define the config arguments accepted and parsed from the command line.

    Returns:
      model_type (String), name (String), config (Dict of config values).
    """
    parser = argparse.ArgumentParser()

    # Register the invariant "model_type" and "name" arguments manually
    parser.add_argument('model_type',
                        choices=[m for m in model_types],
                        help='The type of model, e.g. parikh, or bilstm.')
    parser.add_argument('name',
                        type=str,
                        help='The (unique) name for the training run.')

    # Register all the args in the base_config
    config = {}
    for key in base_config.keys():
        parser.add_argument(
            '--%s' % key,
            help='Set config.%s' % key,
            type=type(base_config[key]))
        config[key] = base_config[key]

    # Parse the args off the command line to return
    args = parser.parse_args()
    model_type = args.model_type
    name = args.name
    for key in base_config.keys():
        passed_value = eval('args.%s' % key)
        if passed_value:
            config[key] = passed_value
    return model_type, name, config
