import random
import numpy as np


class BatchGenerator:
    """Base class for a BatchGenerator.

    The main functions that needs to be implemented to implement the expected
    interface are:
      next_batch()
      batches_per_epoch()
    """

    def __init__(self, batch_size, data, **kwargs):
        """Create a new Batcher.

        Args:
          batch_size: Integer, the number of samples per batch.
          data: can vary depending on sublcass, usually a generator.
          data_size: Integer, number of samples in the data set.
        """
        self.batch_size = batch_size
        self.data_size = len(data)
        self.batches_per_epoch = self.batches_per_epoch(
            batch_size, self.data_size)
        self.batch_count = 0

    @staticmethod
    def batches_per_epoch(batch_size, data_size):
        """Calculate how many batches per epoch.

        If data_size % batch_size != 0, then the last batch will not be of
        batch_size. It is still counted in the number of batches.

        Args:
          batch_size: Integer, the number of samples per batch.
          data_size: Integer, the number of samples in the data set.

        Returns:
          Integer.
        """
        return int(np.ceil(data_size / batch_size))

    def _is_last_batch(self):
        return self.batch_count == self.batches_per_epoch

    def next_batch(self):
        raise NotImplementedError()


class ShuffleBatchGenerator(BatchGenerator):
    """A Standard no-frills batcher.

    This batcher takes the generated data, loads it all into memory,
    and returns random selections for each batch.
    """

    def __init__(self, batch_size, data, **kwargs):
        """Create a new ShuffleBatcher.

        Args:
          batch_size: Integer, the number of samples per batch.
          data: a List or generator with the data.
        """
        super(ShuffleBatchGenerator, self).\
            __init__(batch_size, data, **kwargs)
        self._data = list(data)

    @staticmethod
    def batch_indices(batch_size, batch_number, last_batch):
        """Get the indices to select the batch from the data.

        NOTE: the batch_number should be 1-based. 1 will be subtracted from it
        here to coerce it to the correct indices. It makes sense to apply this
        logic here since in the batching classes 1-based batch_count variables
        are convenient for comparison with batches_per_epoch to determine
        whether or not we are in the last batch.

        Args:
          batch_size: Integer.
          batch_number: Integer, the number of the current batch so we know
            where we are. This needs to be 1-based, not 0-based.
          last_batch: Boolean, indicating whether this is the last batch for a
            given epoch.
        """
        batch_number -= 1
        if last_batch:
            return batch_size * batch_number, -1
        else:
            return batch_size * batch_number, batch_size * (batch_number + 1)

    def _prepare_epoch_data(self):
        random.shuffle(self._data)

    def next_batch(self):
        if self.batch_count == 0:
            self._prepare_epoch_data()
        self.batch_count += 1
        start, end = self.batch_indices(
            self.batch_size, self.batch_count, self._is_last_batch())
        batch = self._data[start:end]
        if self._is_last_batch():
            self.batch_count = 0
        return batch
