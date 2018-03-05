"""For managing experiment results."""
import numpy as np
import scipy as sp
import scipy.stats
from datetime import datetime


def mean_confidence_interval(data, confidence=0.95):
    """
    https://stackoverflow.com/questions/15033511/
    compute-a-confidence-interval-from-sample-data
    """
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


class Experiment:
    """Wrapper for an experiment."""

    def __init__(self, config):
        # config MUST be a dictionary, not a Config object
        self.name = config['name']
        self.model = config['model']
        self.dataset = config['target']
        self.config = config
        self.results = []
        self.n_results = 0
        self.mean = 0.
        self.var = 0.
        self.max = 0.
        self.ci_lower = 0.
        self.ci_upper = 0.

    def __repr__(self):
        info = 'Experiment %s:' % self.name
        info += '\n\tdataset\t%s' % self.dataset
        info += '\n\tn_results\t%s' % self.n_results
        info += '\n\tmax\t\t%s' % self.max
        info += '\n\tmean\t\t%s' % self.mean
        info += '\n\tvar\t\t%s' % self.var
        info += '\n\tci_upper\t%s' % self.ci_upper
        info += '\n\tci_lower\t%s' % self.ci_lower
        info += '\n\tconfig:'
        for key in sorted(self.config.keys()):
            info += '\n\t\t%s\t%s%s' % (key,
                                        '\t' if len(key) < 8 else '',
                                        self.config[key])
        return info

    def consistent(self, config):
        """Checks if a config is consistent with this experiment.
        Args:
          config: Dictionary.

        Returns:
          Bool.
        """
        inconsistencies = []
        # Check for differences in the keysets
        local_keys = set(self.config.keys())
        candidate_keys = set(config.keys())
        do_not_evaluate = ['seed', 'n_runs']
        for x in local_keys.difference(candidate_keys):
            if x not in do_not_evaluate:
                inconsistencies.append('%s not in candidate config.' % x)
        for x in candidate_keys.difference(local_keys):
            if x not in do_not_evaluate:
                inconsistencies.append('%s not in local config.' % x)
        # If any inconsistencies here just return already
        if len(inconsistencies) > 0:
            for i in inconsistencies:
                print(i)
            return False
        # Otherwise perform key-wise comparison
        for key in [k for k in config.keys() if k not in do_not_evaluate]:
            if self.config[key] != config[key]:
                inconsistencies.append('%s != %s for key %s'
                                       % (self.config[key], config[key], key))
        if len(inconsistencies) > 0:
            for i in inconsistencies:
                print(i)
            return False
        return True

    def report(self, run_no, seed, train_acc, tune_acc):
        self.results.append({'run_no': run_no,
                             'seed': seed,
                             'train_acc': train_acc,
                             'tune_acc': tune_acc,
                             'date': datetime.today()})
        self.n_results += 1
        tune_accs = [r['tune_acc'] for r in self.results]
        if self.n_results >= 2:
            self.mean, self.ci_lower, self.ci_upper = mean_confidence_interval(
                tune_accs)
            self.var = np.var(tune_accs)
        self.max = np.max(tune_accs)

    def next_run_no(self):
        return self.n_results + 1


def adapt(json):
    experiment = Experiment(json['config'])
    for key, value in json.items():
        setattr(experiment, key, value)
    return experiment


class ExperimentManager:
    """For loading and saving experiments.

    At the moment this class just loads and saves, but modularizing in this way
    allows for extensibility later. It also mimics the HistoryManager pattern.
    """

    def __init__(self, repo):
        self.repo = repo

    def exists(self, name):
        return self.repo.exists(name=name)

    def load(self, config):
        if not self.repo.exists(name=config['name']):
            info = 'Experiment with name %s not found. Existing names:\n' \
                % config['name']
            for n in [x['name'] for x in self.repo.all()]:
                info += '\t%s\n' % n
            raise ValueError(info)
        json = self.repo.get(name=config['name'])
        experiment = Experiment(json['config'])
        for key, value in json.items():
            setattr(experiment, key, value)
        if not experiment.consistent(config):
            raise ValueError('Existing experiment not consistent with config.')
        return experiment

    def save(self, experiment):
        if self.exists(experiment.name):
            self.repo.update(experiment.__dict__)
        else:
            self.repo.add(experiment.__dict__)
