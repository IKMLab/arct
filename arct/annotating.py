"""Annotations for analysis."""
from arct import DATA_FACTORY as df, PKL
import pandas as pd
import spacy
import numpy as np


def load():
    """Loads the annotations file.

    Returns:
      Dictionary. {dataset: pandas.DataFrame}, where dataset in {train, tune
        test}.
    """
    if PKL.exists('annotations', ['arct']):
        return PKL.load('annotations', ['arct'])
    else:
        return {
            'train': pd.DataFrame({'id': [x['id'] for x in df.train(None)]}),
            'tune': pd.DataFrame({'id': [x['id'] for x in df.tune(None)]}),
            'test': pd.DataFrame({'id': [x['id'] for x in df.test(None)]})
        }


def annotate_negations(annotations):
    print('Annotating negations...')
    train = df.train(None)
    tune = df.tune(None)
    test = df.test(None)
    data = {'train': train, 'tune': tune, 'test': test}
    nlp = spacy.load('en')
    for dname in data.keys():
        print('Working on %s...' % dname)
        has_neg = []
        w0_neg = []
        w1_neg = []
        neg_correct = []
        w0_neg_correct = []
        w1_neg_correct = []
        both_neg = []
        for _, row in annotations[dname].iterrows():
            x = next(y for y in data[dname] if y['id'] == row['id'])
            w0 = any(t.dep_ == 'neg' for t in nlp(x['warrant0']))
            w1 = any(t.dep_ == 'neg' for t in nlp(x['warrant1']))
            has_neg.append(w0 or w1)
            w0_neg.append(w0)
            w1_neg.append(w1)
            neg_correct.append((w0 and x['label'] == 0)
                               or (w1 and x['label'] == 1))
            w0_neg_correct.append(w0 and x['label'] == 0)
            w1_neg_correct.append(w1 and x['label'] == 1)
            both_neg.append(w0 and w1)
        annotations[dname]['has_neg'] = has_neg
        annotations[dname]['w0_neg'] = w0_neg
        annotations[dname]['w1_neg'] = w1_neg
        annotations[dname]['neg_correct'] = neg_correct
        annotations[dname]['w0_neg_correct'] = w0_neg_correct
        annotations[dname]['w1_neg_correct'] = w1_neg_correct
        annotations[dname]['both_neg'] = both_neg
    PKL.save(annotations, 'annotations', ['arct'])


def print_sample(x):
    print('----------')
    print('Id:    \t%s' % x['id'])
    print('Reason:\t%s' % x['reason'])
    print('Claim: \t%s' % x['claim'])
    print('W0:    \t%s' % x['warrant0'])
    print('W1:    \t%s' % x['warrant1'])
    print('Label: \t%s' % x['label'])


def custom(annotations):
    print('Custom annotation interface.')
    print('Commands:')
    print('\tAdd annotations: "@{comma separated list of annotations}')
    print('\tDelete annotations: "-{comma separated list of annotations}')
    print('\tRemove annotation altogether: rm {annotation}')
    print('\tSwitch set: switch {setname}')
    print('\tNext sample: next')
    print('\tLast sample: last')
    print('\tView commands again: help')
    print('\tExit: exit')
    inp = ''
    train = df.train(None)
    tune = df.tune(None)
    test = df.test(None)
    data = {'train': train, 'tune': tune, 'test': test}
    dname = 'test'
    dataset = data['test']
    i = 0
    while inp != 'exit':
        x = dataset[i]
        a = annotations[dname][annotations[dname]['id'] == x['id']]
        tags = [c for c in [n for n in a.columns if n != 'id']
                if int(a[c]) == 1]
        print_sample(x)
        print(', '.join(tags))
        inp = input('')
        if inp.startswith('@'):
            anns = inp.replace('@', '').split(',')
            for ann in anns:
                if ann in annotations[dname].columns:
                    annotations[dname].loc[
                        annotations[dname]['id'] == x['id'], ann] = True
                else:
                    annotations[dname][ann] = np.zeros(len(dataset))
                    annotations[dname].loc[
                        annotations[dname]['id'] == x['id'], ann] = True
            i += 1
            pass
        if inp.startswith('-'):
            anns = inp.replace('-', '').split(',')
            for ann in anns:
                annotations[dname].loc[
                    annotations[dname]['id'] == x['id'], ann] = True
        if inp.startswith('rm'):
            ann = inp.split(' ')[1]
            annotations[dname].drop([ann], axis=1, inplace=True)
        if inp == 'next':
            i += 1
        if inp == 'last':
            i -= 1
        if inp.startswith('switch'):
            dname = inp.split(' ')[1]
            dataset = data[dname]
    PKL.save(annotations, 'annotations', ['arct'])
