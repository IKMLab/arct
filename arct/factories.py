"""For creating training objects from config."""
from ext import factories as ext_factories
from ext import pickling  # need this to avoid circular import
import os
import glovar
import pandas as pd
from arct import collating, models
import random


DATASETS = ['dev-full', 'test-only-data', 'train-full']
TEXT_COLS = ['warrant0', 'warrant1', 'reason', 'claim']
# TODO: add debate title and the like here


class ARCTDataFactory(ext_factories.TextDataFactory):

    def __init__(self):
        super(ARCTDataFactory, self).__init__()

    def data(self, name):
        df = self.df(name)
        has_label = name != 'test-only-data'
        if has_label:
            data = [{'claim': x[1]['claim'],
                     'reason': x[1]['reason'],
                     'warrant0': x[1]['warrant0'],
                     'warrant1': x[1]['warrant1'],
                     'label': x[1]['correctLabelW0orW1'],
                     'id': x[1]['#id'],
                     'debate_title': x[1]['debateTitle'],
                     'debate_info': x[1]['debateInfo']}
                    for x in df.iterrows()]
        else:
            data = [{'claim': x[1]['claim'],
                     'reason': x[1]['reason'],
                     'warrant0': x[1]['warrant0'],
                     'warrant1': x[1]['warrant1'],
                     'id': x[1]['#id'],
                     'debate_title': x[1]['debateTitle'],
                     'debate_info': x[1]['debateInfo']}
                    for x in df.iterrows()]
        return data

    @staticmethod
    def df(name):
        if name not in DATASETS:
            raise ValueError('Unexpected dataset name %r' % name)
        file_path = os.path.join(glovar.ARCT_DIR, name + '.txt')
        if not os.path.exists(file_path):
            raise Exception('No file at %r' % file_path)
        return pd.read_table(file_path)

    def embeddings(self, config):
        return pickling.load(glovar.DATA_DIR, config['embed_type'], ['arct'])

    def test(self, config):
        test_data = self.data('test-only-data')
        test_labels = self.test_labels()
        for x in test_data:
            x['label'] = test_labels[x['id']]
        return test_data

    def test_labels(self):
        labels_path = os.path.join(glovar.ARCT_DIR, 'test-labels.txt')
        data = {}
        with open(labels_path, 'r') as f:
            for line in f.readlines():
                id, label = line.split('\t')
                data[id] = int(label.strip())
        return data

    def train(self, config):
        if config['target'] == 'negs':
            return pickling.load(glovar.DATA_DIR, 'train_negs', ['arct'])
        else:
            data = self.data('train-full')
        if config['train_subsample'] > 0:
            return random.sample(data, config['train_subsample'])
        else:
            return data

    def tune(self, config):
        return self.data('dev-full')

    def vocab(self, config):
        return pickling.load(glovar.DATA_DIR, 'vocab', ['arct'])

    def wordset(self, tokenizer):
        data = self.train(None) + self.tune(None) + self.data('test-only-data')
        text_per_sample = [' '.join([d[c] for c in TEXT_COLS]) for d in data]
        all_text = ' '.join(text_per_sample)
        return set(tokenizer(all_text))


class ARCTTrainFactory(ext_factories.TextTrainFactory):

    def __init__(self, data_factory, history_manager, experiment_manager):
        super(ARCTTrainFactory, self).__init__(
            data_factory, history_manager, experiment_manager)

    def collator(self, config):
        sent_collator = super(ARCTTrainFactory, self).collator(config)
        return collating.ARCTCollator(sent_collator)

    def model(self, config, run_num):
        run_name = config['name'] + str(run_num)
        embeddings = self.data_factory.embeddings(config)
        return models.model(run_name, config, embeddings)
