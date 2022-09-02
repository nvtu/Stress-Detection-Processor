import __init__
from models.classifiers import BinaryStressClassifier
import argparse
import mlflow
mlflow.set_tracking_uri('http://localhost:5010')


parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str)
parser.add_argument("--user_id", type=str, default=None)
parser.add_argument('--model_name', type=str, default='random_forest')
parser.add_argument('--detector_type', type=str, default='independent')
parser.add_argument('--window_shift', type=float, default=0.25)
parser.add_argument('--window_size', type=int, default=60)

args = parser.parse_args()



if __name__ == '__main__':
    clf = BinaryStressClassifier(args.dataset_name, args.model_name, args.detector_type,
            window_shift = args.window_shift, 
            window_size = args.window_size)
    clf.train(test_size = 0.2)