import os
import sys

data_lib = os.path.abspath('../data_processing')
if data_lib not in sys.path:
    sys.path.append(data_lib)


import argparse
import numpy as np
from datapath_manager import *


parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str)
parser.add_argument("--window_shift", type=float, default=0.25)
parser.add_argument("--window_size", type=float, default=60)
args = parser.parse_args()


def combine_features(dataset_name: str):
    """
    Combine statistical features for training
    The function iterates through the statistical feature files in a folder and combines them
    The iteration order follows the lexicographical order of the file names
    NOTE: Check the configuration files of the BranchingNN class to see which features are used and ensure the correct dimensions are used for each branch
    """
    ds_path_manager = get_datapath_manager(dataset_name)
    stats_feature_folder_path = os.path.join(ds_path_manager.stats_feature_path, f'{args.window_size}_{args.window_shift}')
    features_path = sorted([os.path.join(stats_feature_folder_path, feature_path) for feature_path in os.listdir(stats_feature_folder_path)])
    output_folder_path = os.path.join(ds_path_manager.combined_stats_feature_path, f'{args.window_size}_{args.window_shift}')
    create_folder(output_folder_path)
    output_file_path = os.path.join(output_folder_path, f'{dataset_name}_combined_features.npy')
    features = None
    for i, path in enumerate(features_path):
        if i == 0:
            features = np.load(path)
        else:
            feat = np.load(path)
            features = np.concatenate((features, feat), axis=1)
    np.save(output_file_path, features)


if __name__ == '__main__':
    combine_features(args.dataset_name)