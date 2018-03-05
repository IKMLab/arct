"""Models tried for the competition."""
import torch
from torch import nn
from ext import models as ext_models
from ext import encoders as ext_encoders
from ext import layers as ext_layers
from torch import optim
from arct import transfer
import torch.nn.functional as F
from torch.autograd import Variable


def model(name, config, embeddings):
    if config['model'] == 'comp':
        return Comp(name, config, embeddings)
    elif config['model'] == 'lin':
        return LinearClassifier(name, config, embeddings)
    elif config['model'] == 'compc':
        return CompCorr(name, config, embeddings)
    elif config['model'] == 'comprw':
        return CompRW(name, config, embeddings)
    elif config['model'] == 'compbce':
        return CompBCE(name, config, embeddings)
    elif config['model'] == 'compmlp':
        return CompMLP(name, config, embeddings)
    elif config['model'] == 'comprw2':
        return CompRW2(name, config, embeddings)
    else:
        raise ValueError('Unexpected model %r' % config['model'])


class ARCTModel(ext_models.TextModel):
    """Base model."""

    def __init__(self, name, config, embeddings):
        super(ARCTModel, self).__init__(name, config, embeddings)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        if torch.cuda.is_available():
            labels = Variable(                   # labels have to be  LongTensor
                torch.LongTensor(batch.labels),  # for CrossEntropy and Variable
                requires_grad=False).cuda()      # for the accuracy function.
        else:
            labels = Variable(torch.LongTensor(batch.labels),
                              requires_grad=False)

        logits = self.logits(batch)
        loss = self.loss(logits, labels)
        preds = self.predictions(logits)
        acc = self.accuracy(preds, labels)
        return preds, loss, acc

    def loss(self, logits, labels):
        return self.criterion(logits, labels)


class Comp(ARCTModel):
    """Encoder -> compose arg + learn features -> ind. linear classification"""

    def __init__(self, name, config, embeddings):
        super(Comp, self).__init__(name, config, embeddings)

        self.encoder = ext_encoders.LSTMEncoder(config)
        self.pooling = ext_layers.MaxPooling(dim=1)
        self.composition = nn.Linear(self.encoder.size * 4, config.hidden_size)
        self.features = nn.Linear(self.encoder.size * 2, config.hidden_size)
        self.drop = nn.Dropout(p=config['p_drop'])
        self.classifier = nn.Linear(config.hidden_size * 2, 1)

        if config['transfer']:
            print('Transferring encoder params...')
            transfer.load_params(self.encoder, config)

        self.xavier_uniform(self.composition.weight, 'relu')
        self.xavier_uniform(self.features.weight, 'relu')
        self.xavier_uniform(self.classifier.weight, 'none')

        if config['tune_embeds']:
            self.add_param_group(params=self.embeds.parameters(),
                                 name='embeddings',
                                 lr=self.lr * self.emb_lr_factor)
        if config['tune_encoder']:
            self.add_param_group(params=self.encoder.encoder.parameters(),
                                 name='encoder',
                                 lr=self.lr * self.enc_lr_factor)
        self.add_param_group(self.composition.parameters(), 'composition')
        self.add_param_group(self.features.parameters(), 'features')
        self.add_param_group(self.classifier.parameters(), 'classifier')
        self.optimizer = optim.Adam(
            self.param_groups, lr=config['lr'], weight_decay=config['l2'])

    def logits(self, batch):
        # Embedding lookup
        batch.claims.sents = self.lookup(batch.claims.sents, rnn=True)
        batch.reasons.sents = self.lookup(batch.reasons.sents, rnn=True)
        batch.w0s.sents = self.lookup(batch.w0s.sents, rnn=True)
        batch.w1s.sents = self.lookup(batch.w1s.sents, rnn=True)

        # Encoding
        claims = self.encoder(batch.claims)
        reasons = self.encoder(batch.reasons)
        w0s = self.encoder(batch.w0s)
        w1s = self.encoder(batch.w1s)

        # Max pooling
        claims = self.pooling(claims)
        reasons = self.pooling(reasons)
        w0s = self.pooling(w0s)
        w1s = self.pooling(w1s)

        # Regularization
        claims = self.drop(claims)
        reasons = self.drop(reasons)
        w0s = self.drop(w0s)
        w1s = self.drop(w1s)

        # Compose "argument"
        args = F.relu(self.composition(torch.cat([claims, reasons], dim=1)))
        # Learn features for warrants
        w0s = F.relu(self.features(w0s))
        w1s = F.relu(self.features(w1s))

        # Regularization
        args = self.drop(args)
        w0s = self.drop(w0s)
        w1s = self.drop(w1s)

        # Concat arg-warrant pairs for independent classification
        args_w0s = torch.cat([args, w0s], dim=1)
        args_w1s = torch.cat([args, w1s], dim=1)

        # Linear classifier
        classification = self.classifier(torch.cat([args_w0s, args_w1s]))

        # Rearrange for loss calculation
        w0_p, w1_p = torch.split(classification, len(batch), 0)
        return torch.cat([w0_p, w1_p], dim=1)


class LinearClassifier(ARCTModel):
    """Encoder -> Concat -> Classifier."""

    def __init__(self, name, config, embeddings):
        super(LinearClassifier, self).__init__(name, config, embeddings)

        self.encoder = ext_encoders.LSTMEncoder(config)
        self.pooling = ext_layers.MaxPooling(dim=1)
        self.drop = nn.Dropout(p=config['p_drop'])
        self.classifier = nn.Linear(config['encoder_size'] * 8, 2)

        if config['transfer']:
            print('Transferring encoder params...')
            transfer.load_params(self.encoder, config)

        self.xavier_uniform(self.classifier.weight, 'none')

        if config['tune_embeds']:
            self.add_param_group(params=self.embeds.parameters(),
                                 name='embeddings',
                                 lr=self.lr * self.emb_lr_factor)
        if config['tune_encoder']:
            self.add_param_group(params=self.encoder.encoder.parameters(),
                                 name='encoder',
                                 lr=self.lr * self.enc_lr_factor)

        self.add_param_group(self.classifier.parameters(), 'classifier')
        self.optimizer = optim.Adam(
            self.param_groups, lr=config['lr'], weight_decay=config['l2'])

    def logits(self, batch):
        # Embedding lookup
        batch.claims.sents = self.lookup(batch.claims.sents, rnn=True)
        batch.reasons.sents = self.lookup(batch.reasons.sents, rnn=True)
        batch.w0s.sents = self.lookup(batch.w0s.sents, rnn=True)
        batch.w1s.sents = self.lookup(batch.w1s.sents, rnn=True)

        # Encoding
        claims = self.encoder(batch.claims)
        reasons = self.encoder(batch.reasons)
        w0s = self.encoder(batch.w0s)
        w1s = self.encoder(batch.w1s)

        # Max pooling
        claims = self.pooling(claims)
        reasons = self.pooling(reasons)
        w0s = self.pooling(w0s)
        w1s = self.pooling(w1s)

        # Regularization
        claims = self.drop(claims)
        reasons = self.drop(reasons)
        w0s = self.drop(w0s)
        w1s = self.drop(w1s)

        # Concatenation
        features = torch.cat([claims, reasons, w0s, w1s], dim=1)

        # Classification
        return self.classifier(features)


class CompCorr(ARCTModel):
    """Encoder -> compose arg + learn features -> corr. linear classification"""

    def __init__(self, name, config, embeddings):
        super(CompCorr, self).__init__(name, config, embeddings)

        self.encoder = ext_encoders.LSTMEncoder(config)
        self.pooling = ext_layers.MaxPooling(dim=1)
        self.composition = nn.Linear(self.encoder.size * 4, config.hidden_size)
        self.features = nn.Linear(self.encoder.size * 2, config.hidden_size)
        self.drop = nn.Dropout(p=config['p_drop'])
        self.classifier = nn.Linear(config.hidden_size * 3, 2)

        if config['transfer']:
            print('Transferring encoder params...')
            transfer.load_params(self.encoder, config)

        self.xavier_uniform(self.composition.weight, 'relu')
        self.xavier_uniform(self.features.weight, 'relu')
        self.xavier_uniform(self.classifier.weight, 'none')

        if config['tune_embeds']:
            self.add_param_group(params=self.embeds.parameters(),
                                 name='embeddings',
                                 lr=self.lr * self.emb_lr_factor)
        if config['tune_encoder']:
            self.add_param_group(params=self.encoder.encoder.parameters(),
                                 name='encoder',
                                 lr=self.lr * self.enc_lr_factor)
        self.add_param_group(self.composition.parameters(), 'composition')
        self.add_param_group(self.features.parameters(), 'features')
        self.add_param_group(self.classifier.parameters(), 'classifier')
        self.optimizer = optim.Adam(
            self.param_groups, lr=config['lr'], weight_decay=config['l2'])

    def logits(self, batch):
        # Embedding lookup
        batch.claims.sents = self.lookup(batch.claims.sents, rnn=True)
        batch.reasons.sents = self.lookup(batch.reasons.sents, rnn=True)
        batch.w0s.sents = self.lookup(batch.w0s.sents, rnn=True)
        batch.w1s.sents = self.lookup(batch.w1s.sents, rnn=True)

        # Encoding
        claims = self.encoder(batch.claims)
        reasons = self.encoder(batch.reasons)
        w0s = self.encoder(batch.w0s)
        w1s = self.encoder(batch.w1s)

        # Max pooling
        claims = self.pooling(claims)
        reasons = self.pooling(reasons)
        w0s = self.pooling(w0s)
        w1s = self.pooling(w1s)

        # Regularization
        claims = self.drop(claims)
        reasons = self.drop(reasons)
        w0s = self.drop(w0s)
        w1s = self.drop(w1s)

        # Compose "argument"
        args = F.relu(self.composition(torch.cat([claims, reasons], dim=1)))
        # Learn features for warrants
        w0s = F.relu(self.features(w0s))
        w1s = F.relu(self.features(w1s))

        # Regularization
        args = self.drop(args)
        w0s = self.drop(w0s)
        w1s = self.drop(w1s)

        # Concat arg-warrant pairs for independent classification
        features = torch.cat([args, w0s, w1s], dim=1)

        # Linear classifier
        return self.classifier(features)


class CompRW(ARCTModel):
    """Comp model that ignores Claims."""

    def __init__(self, name, config, embeddings):
        super(CompRW, self).__init__(name, config, embeddings)

        self.encoder = ext_encoders.LSTMEncoder(config)
        self.pooling = ext_layers.MaxPooling(dim=1)
        self.features = nn.Linear(self.encoder.size * 2, config.hidden_size)
        self.drop = nn.Dropout(p=config['p_drop'])
        self.classifier = nn.Linear(config.hidden_size * 2, 1)

        if config['transfer']:
            print('Transferring encoder params...')
            transfer.load_params(self.encoder, config)

        self.xavier_uniform(self.features.weight, 'relu')
        self.xavier_uniform(self.classifier.weight, 'none')

        if config['tune_embeds']:
            self.add_param_group(params=self.embeds.parameters(),
                                 name='embeddings',
                                 lr=self.lr * self.emb_lr_factor)
        if config['tune_encoder']:
            self.add_param_group(params=self.encoder.encoder.parameters(),
                                 name='encoder',
                                 lr=self.lr * self.enc_lr_factor)
        self.add_param_group(self.features.parameters(), 'features')
        self.add_param_group(self.classifier.parameters(), 'classifier')
        self.optimizer = optim.Adam(
            self.param_groups, lr=config['lr'], weight_decay=config['l2'])

    def logits(self, batch):
        # Embedding lookup
        batch.reasons.sents = self.lookup(batch.reasons.sents, rnn=True)
        batch.w0s.sents = self.lookup(batch.w0s.sents, rnn=True)
        batch.w1s.sents = self.lookup(batch.w1s.sents, rnn=True)

        # Encoding
        reasons = self.encoder(batch.reasons)
        w0s = self.encoder(batch.w0s)
        w1s = self.encoder(batch.w1s)

        # Max pooling
        reasons = self.pooling(reasons)
        w0s = self.pooling(w0s)
        w1s = self.pooling(w1s)

        # Regularization
        reasons = self.drop(reasons)
        w0s = self.drop(w0s)
        w1s = self.drop(w1s)

        # Learn features
        reasons = F.relu(self.features(reasons))
        w0s = F.relu(self.features(w0s))
        w1s = F.relu(self.features(w1s))

        # Regularization
        reasons = self.drop(reasons)
        w0s = self.drop(w0s)
        w1s = self.drop(w1s)

        # Concat arg-warrant pairs for independent classification
        args_w0s = torch.cat([reasons, w0s], dim=1)
        args_w1s = torch.cat([reasons, w1s], dim=1)

        # Linear classifier
        classification = self.classifier(torch.cat([args_w0s, args_w1s]))

        # Rearrange for loss calculation
        w0_p, w1_p = torch.split(classification, len(batch), 0)
        return torch.cat([w0_p, w1_p], dim=1)


class CompBCE(ARCTModel):
    """Encoder -> compose arg + learn features -> ind. linear classification"""

    def __init__(self, name, config, embeddings):
        super(CompBCE, self).__init__(name, config, embeddings)

        self.encoder = ext_encoders.LSTMEncoder(config)
        self.pooling = ext_layers.MaxPooling(dim=1)
        self.composition = nn.Linear(self.encoder.size * 4, config.hidden_size)
        self.features = nn.Linear(self.encoder.size * 2, config.hidden_size)
        self.drop = nn.Dropout(p=config['p_drop'])
        self.classifier = nn.Linear(config.hidden_size * 2, 1)

        if config['transfer']:
            print('Transferring encoder params...')
            transfer.load_params(self.encoder, config)

        self.xavier_uniform(self.composition.weight, 'relu')
        self.xavier_uniform(self.features.weight, 'relu')
        self.xavier_uniform(self.classifier.weight, 'none')

        self.criterion = nn.BCEWithLogitsLoss()

        if config['tune_embeds']:
            self.add_param_group(params=self.embeds.parameters(),
                                 name='embeddings',
                                 lr=self.lr * self.emb_lr_factor)
        if config['tune_encoder']:
            self.add_param_group(params=self.encoder.encoder.parameters(),
                                 name='encoder',
                                 lr=self.lr * self.enc_lr_factor)
        self.add_param_group(self.composition.parameters(), 'composition')
        self.add_param_group(self.features.parameters(), 'features')
        self.add_param_group(self.classifier.parameters(), 'classifier')
        self.optimizer = optim.Adam(
            self.param_groups, lr=config['lr'], weight_decay=config['l2'])

    def logits(self, batch):
        # Embedding lookup
        batch.claims.sents = self.lookup(batch.claims.sents, rnn=True)
        batch.reasons.sents = self.lookup(batch.reasons.sents, rnn=True)
        batch.w0s.sents = self.lookup(batch.w0s.sents, rnn=True)
        batch.w1s.sents = self.lookup(batch.w1s.sents, rnn=True)

        # Encoding
        claims = self.encoder(batch.claims)
        reasons = self.encoder(batch.reasons)
        w0s = self.encoder(batch.w0s)
        w1s = self.encoder(batch.w1s)

        # Max pooling
        claims = self.pooling(claims)
        reasons = self.pooling(reasons)
        w0s = self.pooling(w0s)
        w1s = self.pooling(w1s)

        # Regularization
        claims = self.drop(claims)
        reasons = self.drop(reasons)
        w0s = self.drop(w0s)
        w1s = self.drop(w1s)

        # Compose "argument"
        args = F.relu(self.composition(torch.cat([claims, reasons], dim=1)))
        # Learn features for warrants
        w0s = F.relu(self.features(w0s))
        w1s = F.relu(self.features(w1s))

        # Regularization
        args = self.drop(args)
        w0s = self.drop(w0s)
        w1s = self.drop(w1s)

        # Concat arg-warrant pairs for independent classification
        args_w0s = torch.cat([args, w0s], dim=1)
        args_w1s = torch.cat([args, w1s], dim=1)

        # Linear classifier
        classification = self.classifier(torch.cat([args_w0s, args_w1s]))

        # Rearrange for loss calculation
        w0_p, w1_p = torch.split(classification, len(batch), 0)
        return torch.cat([w0_p, w1_p], dim=0)

    def loss(self, logits, labels):
        w0_labels = labels == 0
        w1_labels = labels == 1
        labels = torch.cat([w0_labels, w1_labels], dim=0).float()
        return self.criterion(logits.squeeze(), labels)

    def predictions(self, logits):
        half = int(logits.size()[0] / 2)
        logits = torch.cat([logits[0:half].unsqueeze(dim=1),
                            logits[half:].unsqueeze(dim=1)],
                           dim=1).squeeze()
        return super(CompBCE, self).predictions(logits)


class CompMLP(ARCTModel):
    """Encoder -> compose arg + learn features -> ind. linear classification"""

    def __init__(self, name, config, embeddings):
        super(CompMLP, self).__init__(name, config, embeddings)

        self.encoder = ext_encoders.LSTMEncoder(config)
        self.pooling = ext_layers.MaxPooling(dim=1)
        self.composition = nn.Linear(self.encoder.size * 4, config.hidden_size)
        self.features = nn.Linear(self.encoder.size * 2, config.hidden_size)
        self.drop = nn.Dropout(p=config['p_drop'])
        self.mlp = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)

        if config['transfer']:
            print('Transferring encoder params...')
            transfer.load_params(self.encoder, config)

        self.xavier_uniform(self.composition.weight, 'relu')
        self.xavier_uniform(self.features.weight, 'relu')
        self.xavier_uniform(self.classifier.weight, 'relu')
        self.xavier_uniform(self.classifier.weight, 'none')

        if config['tune_embeds']:
            self.add_param_group(params=self.embeds.parameters(),
                                 name='embeddings',
                                 lr=self.lr * self.emb_lr_factor)
        if config['tune_encoder']:
            self.add_param_group(params=self.encoder.encoder.parameters(),
                                 name='encoder',
                                 lr=self.lr * self.enc_lr_factor)
        self.add_param_group(self.composition.parameters(), 'composition')
        self.add_param_group(self.features.parameters(), 'features')
        self.add_param_group(self.mlp.parameters(), 'mlp')
        self.add_param_group(self.classifier.parameters(), 'classifier')
        self.optimizer = optim.Adam(
            self.param_groups, lr=config['lr'], weight_decay=config['l2'])

    def logits(self, batch):
        # Embedding lookup
        batch.claims.sents = self.lookup(batch.claims.sents, rnn=True)
        batch.reasons.sents = self.lookup(batch.reasons.sents, rnn=True)
        batch.w0s.sents = self.lookup(batch.w0s.sents, rnn=True)
        batch.w1s.sents = self.lookup(batch.w1s.sents, rnn=True)

        # Encoding
        claims = self.encoder(batch.claims)
        reasons = self.encoder(batch.reasons)
        w0s = self.encoder(batch.w0s)
        w1s = self.encoder(batch.w1s)

        # Max pooling
        claims = self.pooling(claims)
        reasons = self.pooling(reasons)
        w0s = self.pooling(w0s)
        w1s = self.pooling(w1s)

        # Regularization
        claims = self.drop(claims)
        reasons = self.drop(reasons)
        w0s = self.drop(w0s)
        w1s = self.drop(w1s)

        # Compose "argument"
        args = F.relu(self.composition(torch.cat([claims, reasons], dim=1)))
        # Learn features for warrants
        w0s = F.relu(self.features(w0s))
        w1s = F.relu(self.features(w1s))

        # Regularization
        args = self.drop(args)
        w0s = self.drop(w0s)
        w1s = self.drop(w1s)

        # Concat arg-warrant pairs for independent classification
        args_w0s = torch.cat([args, w0s], dim=1)
        args_w1s = torch.cat([args, w1s], dim=1)

        # MLP
        features = F.relu(self.mlp(torch.cat([args_w0s, args_w1s])))

        # Final Regularization
        features = self.drop(features)

        # Linear classifier
        classification = self.classifier(features)

        # Rearrange for loss calculation
        w0_p, w1_p = torch.split(classification, len(batch), 0)
        return torch.cat([w0_p, w1_p], dim=1)


class CompRW2(ARCTModel):
    """Comp model that ignores Claims."""

    def __init__(self, name, config, embeddings):
        super(CompRW2, self).__init__(name, config, embeddings)

        self.encoder = ext_encoders.LSTMEncoder(config)
        self.pooling = ext_layers.MaxPooling(dim=1)
        self.U = nn.Linear(self.encoder.size * 2, config.hidden_size)
        self.features = nn.Linear(self.encoder.size * 2, config.hidden_size)
        self.drop = nn.Dropout(p=config['p_drop'])
        self.classifier = nn.Linear(config.hidden_size * 2, 1)

        if config['transfer']:
            print('Transferring encoder params...')
            transfer.load_params(self.encoder, config)

        self.xavier_uniform(self.features.weight, 'relu')
        self.xavier_uniform(self.classifier.weight, 'none')

        if config['tune_embeds']:
            self.add_param_group(params=self.embeds.parameters(),
                                 name='embeddings',
                                 lr=self.lr * self.emb_lr_factor)
        if config['tune_encoder']:
            self.add_param_group(params=self.encoder.encoder.parameters(),
                                 name='encoder',
                                 lr=self.lr * self.enc_lr_factor)
        self.add_param_group(self.features.parameters(), 'features')
        self.add_param_group(self.classifier.parameters(), 'classifier')
        self.optimizer = optim.Adam(
            self.param_groups, lr=config['lr'], weight_decay=config['l2'])

    def logits(self, batch):
        # Embedding lookup
        batch.reasons.sents = self.lookup(batch.reasons.sents, rnn=True)
        batch.w0s.sents = self.lookup(batch.w0s.sents, rnn=True)
        batch.w1s.sents = self.lookup(batch.w1s.sents, rnn=True)

        # Encoding
        reasons = self.encoder(batch.reasons)
        w0s = self.encoder(batch.w0s)
        w1s = self.encoder(batch.w1s)

        # Max pooling
        reasons = self.pooling(reasons)
        w0s = self.pooling(w0s)
        w1s = self.pooling(w1s)

        # Regularization
        reasons = self.drop(reasons)
        w0s = self.drop(w0s)
        w1s = self.drop(w1s)

        # Learn features
        reasons = F.relu(self.U(reasons))
        w0s = F.relu(self.features(w0s))
        w1s = F.relu(self.features(w1s))

        # Regularization
        reasons = self.drop(reasons)
        w0s = self.drop(w0s)
        w1s = self.drop(w1s)

        # Concat arg-warrant pairs for independent classification
        args_w0s = torch.cat([reasons, w0s], dim=1)
        args_w1s = torch.cat([reasons, w1s], dim=1)

        # Linear classifier
        classification = self.classifier(torch.cat([args_w0s, args_w1s]))

        # Rearrange for loss calculation
        w0_p, w1_p = torch.split(classification, len(batch), 0)
        return torch.cat([w0_p, w1_p], dim=1)
