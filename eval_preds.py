from arct import DATA_FACTORY
import os
import glovar
import pandas as pd


print('Evaluating all prediction files against test labels...')
test_labels = DATA_FACTORY.test_labels()
dir_path = os.path.join(glovar.DATA_DIR, 'predictions')
file_names = []
results = []
for file_name in sorted([n for n in os.listdir(dir_path) if 'test' in n]):
    print('Evaluating %s...' % file_name)
    with open(os.path.join(dir_path, file_name)) as f:
        n = 0.
        correct = 0.
        for line in f.readlines():
            n += 1.
            id, label = line.split('\t')
            label = int(label.strip())
            if label == test_labels[id]:
                correct += 1.
    result = correct / n
    print('%6.4f' % result)
    file_names.append(file_name)
    results.append(result)
df = pd.DataFrame({'file_name': file_names, 'result': results})
df.to_csv(os.path.join(glovar.DATA_DIR, 'test_evals.csv'))
