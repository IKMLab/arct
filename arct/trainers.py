from ext import tensor_flow, pickling
from arct import dbi
import spacy
import numpy as np


class TFTrainer(tensor_flow.TensorFlowTrainer):
    def __init__(self, model, history, train_data, tune_data, ckpt_dir):
        super(TFTrainer, self).__init__(
            model, history, train_data, tune_data, ckpt_dir, dbi)
        self.nlp = spacy.load('en')

    def feed_dict(self, batch):
        # get the specific bits of data
        reasons = [s[0] for s in batch]
        claims = [s[1] for s in batch]
        warrants = [s[2] for s in batch]
        alt_warrants = [s[3] for s in batch]
        labels = np.array([s[4] for s in batch], dtype='int32')
        labels = labels.reshape((1, len(batch)))

        # tokenize the sentences
        reasons = [self.nlp(s) for s in reasons]
        claims = [self.nlp(s) for s in claims]
        warrants = [self.nlp(s) for s in warrants]
        alt_warrants = [self.nlp(s) for s in alt_warrants]

        # lookup the indices for each token
        reasons = [[self.model.vocab_dict[t.text] for t in s]
                   for s in reasons]
        claims = [[self.model.vocab_dict[t.text] for t in s]
                  for s in claims]
        warrants = [[self.model.vocab_dict[t.text] for t in s]
                    for s in warrants]
        alt_warrants = [[self.model.vocab_dict[t.text] for t in s]
                        for s in alt_warrants]

        # pad the sequences
        all_sents = [s for sents in [reasons, claims, warrants, alt_warrants]
                     for s in sents]
        max_seq_len = max([len(s) for s in all_sents])
        for x in [reasons, claims, warrants, alt_warrants]:
            for seq in x:
                if len(seq) < max_seq_len:
                    for _ in range(max_seq_len - len(seq)):
                        seq.append(0)

        # get the feed dict
        feed_dict = self.model.dropout_feed_dict()
        feed_dict[self.model.r] = reasons
        feed_dict[self.model.c] = claims
        feed_dict[self.model.w] = warrants
        feed_dict[self.model.aw] = alt_warrants
        feed_dict[self.model.labels] = labels
        return feed_dict

    def predict(self, batch):
        feed_dict = self.feed_dict(batch)
        acc = self.sess.run(self.model.accuracy, feed_dict=feed_dict)
        return acc

    def step(self, batch):
        feed_dict = self.feed_dict(batch)
        loss, acc, _ = self.sess.run([self.model.loss,
                                      self.model.accuracy,
                                      self.model.optimize],
                                     feed_dict=feed_dict)
        return loss, acc
