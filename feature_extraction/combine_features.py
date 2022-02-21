import os
import sys

data_lib = os.path.abspath('../data_preprocessing')
if data_lib not in sys.path:
    sys.path.append(data_lib)


import argparse
import numpy as np
from datapath_manager import *


parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str)
args = parser.parse_args()


def combine_features(dataset_name: str):
    ds_path_manager = get_datapath_manager(dataset_name)
    features_path = sorted([os.path.join(ds_path_manager.stats_feature_path, feature_path) for feature_path in os.listdir(ds_path_manager.stats_feature_path)])
    output_file_path = ds_path_manager.combined_stats_feature_path
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