import os
import sys

data_lib = os.path.abspath('../data_processing')
if data_lib not in sys.path:
    sys.path.append(data_lib)


import argparse
import numpy as np
from datapath_manager import DataPathManager, create_folder

"""
This file is created to combine the statistical features for training
"""

class FeatureCombiner:

    def __init__(self, dataset_name: str, window_shift: float = 0.25, window_size: int = 60):
        self.dataset_name = dataset_name
        self.window_shift = window_shift
        self.window_size = window_size

        self.ds_path_manager = DataPathManager(self.dataset_name).ds_path_manager
        

    def combine_features(self):
        """
        Combine statistical features for training
        The function iterates through the statistical feature files in a folder and combines them
        The iteration order follows the lexicographical order of the file names
        NOTE: Check the configuration files of the BranchNeuralNetwork class to see which features are used and ensure the correct dimensions are used for each branch
        """

        stats_feature_folder_path = os.path.join(self.ds_path_manager.stats_feature_path, f'{self.window_size}_{self.window_shift}')
        features_path = sorted([os.path.join(stats_feature_folder_path, feature_path) for feature_path in os.listdir(stats_feature_folder_path)])
        output_folder_path = os.path.join(self.ds_path_manager.combined_stats_feature_path, f'{self.window_size}_{self.window_shift}')
        create_folder(output_folder_path)
        output_file_path = os.path.join(output_folder_path, f'{self.dataset_name}_combined_features.npy')
        features = None
        for i, path in enumerate(features_path):
            if i == 0:
                features = np.load(path)
            else:
                feat = np.load(path)
                features = np.concatenate((features, feat), axis=1)
        np.save(output_file_path, features)


    def combined_features_for_user(self, user_id: str):
        """
        Combine statistical features for training
        The function iterates through the statistical feature files in a folder and combines them
        The iteration order follows the lexicographical order of the file names
        NOTE: Check the configuration files of the BranchNeuralNetwork class to see which features are used and ensure the correct dimensions are used for each branch
        """

        stats_feature_folder_path = os.path.join(self.ds_path_manager.user_data_paths[user_id].feature_path, f'{self.window_size}_{self.window_shift}')
        features_path = sorted([os.path.join(stats_feature_folder_path, feature_path) for feature_path in os.listdir(stats_feature_folder_path)])
        output_folder_path = os.path.join(self.ds_path_manager.user_data_paths[user_id].combined_stats_feature_path, f'{self.window_size}_{self.window_shift}')
        create_folder(output_folder_path)
        output_file_path = os.path.join(output_folder_path, f'{user_id}_combined_features.npy')
        features = None
        for i, path in enumerate(features_path):
            if i == 0:
                features = np.load(path)
            else:
                feat = np.load(path)
                features = np.concatenate((features, feat), axis=1)
        np.save(output_file_path, features)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--window_shift", type=float, default=0.25)
    parser.add_argument("--window_size", type=float, default=60)
    parser.add_argument("--user_id", type=str, default=None)

    args = parser.parse_args()

    feature_combiner = FeatureCombiner(args.dataset_name, args.window_shift, args.window_size)
    if args.user_id is None:
        feature_combiner.combine_features()
    else:
        feature_combiner.combine_features_for_user(args.user_id)