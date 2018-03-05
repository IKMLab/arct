from ext import pickling, training, experiments, gridsearch
import glovar
from arct import mongoi, factories


PKL = pickling.Pickler(glovar.DATA_DIR)
DBI = mongoi.MongoDbInterface()
DATA_FACTORY = factories.ARCTDataFactory()
HISTORY_MANAGER = training.HistoryManager(DBI.histories)
EXPERIMENT_MANAGER = experiments.ExperimentManager(DBI.experiments)
TRAIN_FACTORY = factories.ARCTTrainFactory(
    DATA_FACTORY, HISTORY_MANAGER, EXPERIMENT_MANAGER)
GRID_MANAGER = gridsearch.GridManager(DBI.grid)
