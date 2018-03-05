"""For conducting grid search on parameters."""
import torch
import itertools


class GridSearch:
    """Wrapper for grid search.

    The "name" attribute defines a unique search. If we have any record with the
    same name already saved, all subsequent configs will follow that when
    restarting.

    Each run through a combination of parameters is not considered an experiment
    but rather we just want to keep a record of the name, config, and result. We
    ultimately want to return the best config, not evaluate it as an experiment.
    """

    def __init__(self, config, search_space, processor, manager):
        """Create a new GridSearch.

        Args:
          config: Dictionary, for defining static config parameters not in the
            search space.
          search_space: Dictionary of parameter names and values to search over.
          processor: callable to train and return best tuning accuracy.
          manager: GridManager.

        Raises:
          ValueError: if there is an inconsistency with the base config and any
            existing record of the same name.
          ValueError: if "seed" not specified on config.
        """
        config['override'] = True
        inconsistency, keys = manager.inconsistent(config, search_space.keys())
        if inconsistency:
            self.print_config(config)
            print('Inconsistent keys:')
            print(keys)
            raise ValueError('Inconsistent config for grid. Details above.')
        if config['seed'] < 1:
            raise ValueError('Grid requires a valid seed - saw %s'
                             % config['seed'])
        self.base_config = config
        self.processor = processor
        self.search_space = search_space
        self.manager = manager
        self.keys, self.combos = self.combinations(search_space)
        self.n_combos = len(self.combos)

    def __call__(self):
        self.search()

    @staticmethod
    def combinations(search_space):
        """Get combinations of the parameters in the space for grid search.

        Args:
          search_space: Dictionary. The keys are the hyperparameter names and
            the values are lists of candidate values to search over.

        Returns:
          keys: List of hyperparameter names in the order in which they appear
            in the combos.
          combos: List of tuples of combined hyperparameter values for search.
        """
        keys = search_space.keys()
        dim_values = [dims for dims in search_space.values()]
        combos = list(itertools.product(*dim_values))
        return keys, combos

    @staticmethod
    def print_config(config):
        for key in sorted(config.keys()):
            print('%s\t%s%s' % (key, '\t' if len(key) < 8 else '', config[key]))

    def search(self):
        """Perform grid search.

        Returns:
          max_score: Float.
          best_params: Dictionary.
        """
        print('Performing grid search for %s...' % self.base_config['name'])
        for i, combo in enumerate(self.combos):
            params = dict(zip(self.keys, combo))
            config = self.base_config.copy()
            for key, val in params.items():
                config[key] = val
            print(config)
            if self.manager.outstanding(config):  # don't re-run a combo
                print('Evaluating combination %s/%s..' % (i + 1, self.n_combos))
                torch.manual_seed(self.base_config['seed'])
                result = self.processor.run_search(config, i)
                self.manager.report(config=config, result=result)
        max_score, best_config = self.manager.best(self.base_config['name'])
        print('Search complete.')
        print('Max score: %s' % max_score)
        print('Attaining configs:')
        self.print_config(best_config)


class GridManager:
    """For managing results: saving, loading, querying.

    Default uses MongoDB.
    """

    def __init__(self, repo):
        """Create a new GridManager.

        Args:
          repo: hsdbi.mongo.MongoRepository, for access to the collection that
            stores the records.
        """
        self.repo = repo

    def best(self, name):
        """Get best result and config(s).

        Args:
          name: String.

        Returns:
          best_result: Float.
          best_configs: Dictionary.
        """
        records = list(self.repo.search(name=name))
        best_result = max([r['result'] for r in records])
        best_configs = [r for r in records if r['result'] == best_result]
        if len(best_configs) > 1:
            print('Too many best configs, returning None...')
            best_config = None
        else:
            best_config = best_configs[0]
        return best_result, best_config

    def inconsistent(self, config, search_space_keys):
        """Determine if a base_config is inconsistent with existing records.

        Args:
          config: Dictionary.
          search_space_keys: List of strings, the config values participating in
            the search space.

        Returns:
          inconsistency: Bool.
          inconsistent_keys: List of strings.
        """
        existing = list(self.repo.search(name=config['name']))
        inconsistent_keys = []
        if len(existing) > 0:
            x = existing[0]
            for key in [k for k in config.keys() if k not in search_space_keys]:
                if x[key] != config[key]:
                    inconsistent_keys.append(key)
        return len(inconsistent_keys) > 0, inconsistent_keys

    def outstanding(self, config):
        """Determine if a config combination is outstanding and needs to be run.

        Args:
          config: Dictionary.

        Returns:
          Bool.
        """
        return not self.repo.exists(**config)

    def report(self, config, result):
        """Report the result of a new run.

        Args:
          config: Dictionary.
          result: Float.
        """
        self.repo.add(result=result, **config)
