"""Script for generating predictions.


"""
import argparse
from arct import prediction


parser = argparse.ArgumentParser()
parser.add_argument('name',
                    type=str,
                    help='The nam of the training run / experiment')
args = parser.parse_args()
name = getattr(args, 'name')
predictor = prediction.Predictor()
predictor(name)
