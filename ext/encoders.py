"""RNN encoder modules."""
import torch
from torch import nn


#
# Encoders


class Encoder(nn.Module):
    """Base encoder (abstract).

    Deriving classes need to initialize an "encoder" attribute (e.g. with an
    LSTM or GRU or what have you).
    """

    def __init__(self, config):
        """Create a new LSTM encoder.

        Args:
          config: Dictionary. We expect the attributes:
            embed_size: Integer, size of input word embeddings.
            encoder_size: Integer.
            p_drop_rnn: Float, probability of dropping neurons between recurrent
              timesteps.
            encoder_layers: Integer.
            bidirectional: Bool.
        """
        super(Encoder, self).__init__()
        self.embed_size = config['embed_size']
        self.size = config['encoder_size']
        self.p_drop = config['p_drop_rnn']
        self.num_layers = config['encoder_layers']
        self.bidirectional = config['bidirectional']

    def forward(self, sents):
        """Perform forward pass.

        Args:
          sents: ext.collating.RNNSents object. The sents attribute is a 3D
            tensor of shape [timesteps, batch_size, embed_size], with all word
            vectors already looked up. By collating convention for PyTorch RNNs,
            these sentences are already ordered by sentence length. This method
            will reorder them using sents.rev_ix_sort to realign them with their
            labels.

        Returns:
          torch.autograd.Variable of shape [batch_size, hidden_size * 2].
        """
        rnn_outputs = self.rnn_forward(sents)
        outputs = self.rev_sort(sents, rnn_outputs)
        return outputs

    def rnn_forward(self, sents):
        # Pack the sequences for processing
        packed = nn.utils.rnn.pack_padded_sequence(sents.sents, sents.lens)

        # Get the output
        outputs = self.encoder(packed)[0]  # 0 selects the hidden states

        # Unpack the LSTM outputs -> [longest_sent, batch_size, 2 * embed_dim]
        outputs = nn.utils.rnn.pad_packed_sequence(outputs)[0]  # 0 is the seq
                                                                # 1 is indices

        return outputs

    def rev_sort(self, sents, outputs):
        # [batch_size, longest_sent, 2 * embed_dim]
        outputs = outputs.permute([1, 0, 2])

        # Reorder the outputs to line up with the labels
        if torch.cuda.is_available():
            outputs = outputs[torch.LongTensor(sents.rev_ix_sort).cuda()]
        else:
            outputs = outputs[torch.LongTensor(sents.rev_ix_sort)]

        # TODO: is it more sensible to order the labels? In pre-processing even?

        return outputs


class LSTMEncoder(Encoder):
    """LSTM Encoder."""

    def __init__(self, config):
        super(LSTMEncoder, self).__init__(config)
        self.encoder = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)


class GRUEncoder(Encoder):
    """GRU Encoder."""

    def __init__(self, config):
        super(GRUEncoder, self).__init__(config)
        self.encoder = nn.GRU(
            input_size=self.embed_size,
            hidden_size=self.size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
