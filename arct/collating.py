"""Collating ARCT data for training."""


class ARCTBatch:
    """Data structure for a batch of ARCT data points."""

    def __init__(self, claims, reasons, w0s, w1s, labels, ids):
        """Create a new NLIBatch.
        Args:
          claims: ext.collating.RNNSents.
          reasons: ext.collating.RNNSents.
          w0s: ext.collating.RNNSents.
          w1s: ext.collating.RNNSents.
          labels: List of integers.
          ids: List of strings.
        """
        self.claims = claims
        self.reasons = reasons
        self.w0s = w0s
        self.w1s = w1s
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.ids)


class ARCTCollator:
    """Collate function for non-RNNs."""

    def __init__(self, sent_collator):
        self.sent_collator = sent_collator

    def __call__(self, data):
        # data is a list of JSON
        reasons = self.sent_collator([x['reason'] for x in data])
        claims = self.sent_collator([x['claim'] for x in data])
        w0s = self.sent_collator([x['warrant0'] for x in data])
        w1s = self.sent_collator([x['warrant1'] for x in data])
        if 'label' in data[0].keys():
            labels = [x['label'] for x in data]
        else:
            labels = None
        ids = [x['id'] for x in data]
        return ARCTBatch(
            claims=claims,
            reasons=reasons,
            w0s=w0s,
            w1s=w1s,
            labels=labels,
            ids=ids)
