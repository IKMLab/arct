"""For generating predictions on ARCT test set for submission."""
from arct import DBI, TRAIN_FACTORY as TF, DATA_FACTORY as DF, configuration
import os
import glovar
from ext import experiments
import torch


class Predictor:

    def __init__(self):
        pass

    def __call__(self, name):
        print('Generating ARCT predictions...')
        # using the experiment name, we want to generate preds for ALL runs
        experiment = DBI.experiments.get(name=name)
        experiment = experiments.adapt(experiment)
        config = experiment.config
        config = configuration.adapt(config)
        train_data = DF.data('train-full')
        tune_data = DF.data('dev-full')
        test_data = DF.data('test-only-data')
        collator = TF.collator(config)
        train_loader = TF.data_loader(config, train_data, collator)
        tune_loader = TF.data_loader(config, tune_data, collator)
        test_loader = TF.data_loader(config, test_data, collator)
        saver = TF.saver(config)
        for n in range(1, experiment.n_results + 1):
            print('Run number %s...' % n)
            model = TF.model(config, n)
            saver.load(model, is_best=True)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
            predictions = {'train': [], 'tune': [], 'test': []}
            for _, batch in enumerate(train_loader):
                logits = model.logits(batch)
                preds = model.predictions(logits)
                for i in range(len(batch)):
                    predictions['train'].append(
                        (batch.ids[i], preds[i].cpu().data.numpy()[0]))
            for _, batch in enumerate(tune_loader):
                logits = model.logits(batch)
                preds = model.predictions(logits)
                for i in range(len(batch)):
                    predictions['tune'].append(
                        (batch.ids[i], preds[i].cpu().data.numpy()[0]))
            for _, batch in enumerate(test_loader):
                logits = model.logits(batch)
                preds = model.predictions(logits)
                for i in range(len(batch)):
                    predictions['test'].append(
                        (batch.ids[i], preds[i].cpu().data.numpy()[0]))
            train_file_name = os.path.join(
                glovar.DATA_DIR, 'predictions', model.name + '_train.txt')
            tune_file_name = os.path.join(
                glovar.DATA_DIR, 'predictions', model.name + '_tune.txt')
            test_file_name = os.path.join(
                glovar.DATA_DIR, 'predictions', model.name + '_test.txt')
            with open(train_file_name, 'w') as f:
                for id, label in predictions['train']:
                    f.write('%s\t%s\n' % (id, label))
            with open(tune_file_name, 'w') as f:
                for id, label in predictions['tune']:
                    f.write('%s\t%s\n' % (id, label))
            with open(test_file_name, 'w') as f:
                for id, label in predictions['test']:
                    f.write('%s\t%s\n' % (id, label))
        print('Done.')
