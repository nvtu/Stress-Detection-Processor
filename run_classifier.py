from models.ML_Classifiers import BinaryClassifier
from data_preprocessing.dataloader import *
from data_preprocessing.datapath_manager import *
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

    clf = BinaryClassifier(dataset, ground_truth, args.model_name, basic_logo_validation = True, groups = groups)
    results = clf.exec_classifier()
    print('----------------------------------')   

