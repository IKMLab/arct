import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import init
import torch.nn.functional as F


"""
Possibilities:
1) Compare with an LSTM encoder
2) First pass had the "support" concat with Claim, then classified
    -> another option would be to COMPOSE again, using the same params (?)
    -> less params better on smaller data set ?
    -> worth comparing
3) The single vector encoding is a weak technique - attention is best.
   Could I find a way to use attention over tree nodes?
    -> Given the nature of the task and the data, I could run a BiLSTM, concat
       the hidden states with the word vectors, run those through a simplified
       tree composition function, then run attention over those?
    -> Degrees of model complexity: BiLSTM => BiLSTM + Simple Tree => TreeGRU...
"""


class TreeGRU(nn.Module):
    def __init__(self, hidden_size):
        super(TreeGRU, self).__init__()

    def forward(self, tree):
        return 0


class ACCompose(nn.Module):
    def __init__(self, hidden_size, num_layers, drop_rate):
        super(ACCompose, self).__init__()
        self.num_layers = num_layers
        self.fc = dict()
        self.fc[1] = nn.Linear(2 * hidden_size, hidden_size)
        for i in range(2, num_layers + 1):
            self.fc[i] = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(p=drop_rate)
        # self init to xavier with 5/3 gain

    def forward(self, ac1, ac2):
        x = torch.cat([ac1, ac2], 1)
        for i in range(1, self.num_layers + 1):
            z = self.fc[i](x)
            h = F.relu(z)
            x = self.drop(h)
        return x


class ArgCompare(nn.Module):
    def __init__(self, hidden_size, drop_rate):
        super(ArgCompare, self).__init__()
        self.fc1 = nn.Linear(3 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.drop = nn.Dropout(p=drop_rate)
        # init params with xavier with gain 5/3

    def forward(self, rw, raw, c):
        # Thinking about this:
        # I want the weight vector in the last linear layer to be a prototype
        # of an argument that follows. So, instead of concating all three (and
        # also adding more params), process the two candidates separately,
        # comparing them individually to the prototype vector, and take the
        # argmax of the resulting column vector to get the classification.
        # OR: return a reshaped row vector, or 1-d vector is better.
        rw_in = torch.cat([rw, c], 1)
        raw_in = torch.cat([raw, c], 1)
        f_in = torch.cat([rw_in, raw_in], 0)
        h1 = self.drop(F.relu(self.fc1(f_in)))
        h2 = self.drop(F.relu(self.fc2(h1)))
        logits = self.fc3(h2)  # how to reshape here?
        return logits


class TreeGRUModel(nn.Module):
    def __init__(self, hidden_size):
        super(TreeGRUModel, self).__init__()
        # I want the Tree GRU as a component, a module
        # forward should be the encode for a whole sentence
        # will need to pay attention to inputs / data loading
        # I will assume we have the module
        # do the rest of the network now
        self.tree_gru = TreeGRU(hidden_size)
        self.compose = ACCompose(hidden_size, 1, 0.5)
        self.compare = ArgCompare(hidden_size, 0.5)

    def forward(self, batch):
        # tree_gru takes a batch object which has all our wirings figure out
        # it returns encoded vectors for all the components
        # NOTE: given component boundaries, will we experience issues with
        # the dependency parses? This will need to be inspected.
        r, w, aw, c = self.tree_gru(batch)
        # ac_comp composes the argument components into a single "support"
        # vector that the claim should follow from
        rw, raw = self.compose(r, w, aw)
        # eval concatenates the claim and the two candidate support vectors and
        # tells us which one it likes
        logits = self.compare(rw, raw, c)
        return logits

    def forward2(self, batch):
        r, w, aw, c = self.tree_gru(batch)
        rw = self.compose(r, w)
        raw = self.compose(r, aw)
        crw = self.compose(rw, c)
        craw = self.compose(raw, c)
        logits = self.compare(crw, craw)
        return logits
