"""Script for performing grid search."""
from ext import gridsearch, argparsing, processing
import arct
from arct import configuration, gridspace


config = argparsing.parse(configuration.Config())
space = gridspace.space(config)
processor = processing.Processor(arct.TRAIN_FACTORY)
searcher = gridsearch.GridSearch(config, space, processor, arct.GRID_MANAGER)
searcher.search()
