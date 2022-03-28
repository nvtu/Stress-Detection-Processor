import sys
import os

data_lib = os.path.abspath('./data_processing')
if data_lib not in sys.path:
    sys.path.append(data_lib)

from models.ML_Classifiers import BinaryClassifier
from data_processing.dataloader import *
from data_processing.datapath_manager import *
from data_processing.result_utils import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str)
parser.add_argument("--user_id", type=str, default=None)
parser.add_argument('--model_name', type=str, default='random_forest')
parser.add_argument('--detector_type', type=str, default='General')
parser.add_argument('--window_shift', type=float, default=0.25)
parser.add_argument('--window_size', type=float, default=60)
args = parser.parse_args()



if __name__ == '__main__':
    dataloader = DataLoader(args.dataset_name)
    dataset, ground_truth, groups = dataloader.load_data_for_training()
    # ds_path_manager = get_datapath_manager(args.dataset_name)

    clf = BinaryClassifier(dataset, ground_truth, 
        args.model_name, 
        subject_independent = True if args.detector_type == 'General' else False, 
        subject_dependent = True if args.detector_type == 'Personal' else False, 
        groups = groups)
    results = clf.exec_classifier()
    result_helper = ResultUtils(args.dataset_name)
    result_helper.dump_result_to_csv(results, columns=['user_id', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score'])
    print('----------------------------------')   

