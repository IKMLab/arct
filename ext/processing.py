"""Facade bringing all together into a callable, reusable process."""
import torch
import random


class Processor:
    """For processing an experiment run."""

    def __init__(self, factory):
        self.factory = factory

    def __call__(self, config):
        self.run_exp(config)

    def set_seed(self, config):
        if config['seed'] >= 1:
            seed = config['seed']
        else:
            seed = random.choice(range(10000))
        random.seed(seed)
        torch.manual_seed(seed)
        return seed

    def run_search(self, config, combo_num):
        _ = self.set_seed(config)
        trainer = self.factory.trainer(config, combo_num, is_grid=True)
        best_tune_acc = trainer.train()
        trainer.logger.log('Best tuning accuracy: %6.4f' % best_tune_acc)
        return best_tune_acc

    def run_exp(self, config):
        experiment = self.factory.experiment(config)
        for _ in range(config['n_runs']):
            run_num = experiment.n_results + 1
            seed = self.set_seed(config)
            trainer = self.factory.trainer(config, run_num, is_grid=False)
            trainer.logger.log('Running experiment "%s"...' % config['name'])
            trainer.logger.log('Run no. %s' % run_num)
            trainer.logger.log('Seed: %s' % seed)
            trainer.logger.log(config.__repr__())
            best_tune_acc = trainer.train()
            trainer.logger.log('Best tuning accuracy: %6.4f' % best_tune_acc)
            trainer.saver.load(trainer.model, is_best=True)
            train_acc = 0.
            for _, batch in enumerate(trainer.train_loader):
                __, acc = trainer.step(batch)
                train_acc += acc
            train_acc = train_acc / len(trainer.train_loader)
            trainer.logger.log('Corresponding training accuracy: %6.4f'
                               % train_acc)
            trainer.logger.log('\n')
            experiment.report(run_num, seed, train_acc, best_tune_acc)
            self.factory.experiment_manager.save(experiment)
        print(experiment)
