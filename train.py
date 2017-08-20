from arct import bilstm1, data, glovar, trainers, dbi
from ext import parameters, batching, models, histories


# Parse command line args and get history (which wraps config).
params, arg_config = parameters.parse_arguments(glovar.MODELS, models.Config())
if params.override and dbi.history.train.exists(_id=params.name):
    print('Override selected - deleting history "%s"' % params.name)
    dbi.history.train.delete(_id=params.name)
print('Getting or creating History...')
history = histories.History(params.name, models.Config(**arg_config))
config = history.config
print('Config as follows:')
for key in sorted(list(config.keys())):
    print('\t%s \t%s%s' % (key, '\t' if len(key) < 15 else '', config[key]))


# Load the data, embeddings, and vocab
embeddings = data.embedding('glove', '840B', 300)
vocab_dict = data.vocab()
train_data, tune_data = data.train_and_tune_data()


# Create batching objects for data
train_batcher = batching.ShuffleBatchGenerator(config.batch_size, train_data)
tune_batcher = batching.ShuffleBatchGenerator(config.batch_size, tune_data)


# Create the model
model = bilstm1.BiLSTM1(config, vocab_dict, embeddings)


# Create the trainer
trainer = trainers.TFTrainer(
    model, history, train_batcher, tune_batcher, glovar.CKPT_DIR)


# Train
trainer.train()
