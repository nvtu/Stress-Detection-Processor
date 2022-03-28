import os
import sys

parent_dir = os.path.abspath('..')
data_lib = os.path.abspath('../data_processing')
if data_lib not in sys.path:
    sys.path.append(data_lib)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


import argparse
from datapath_manager import *
from dataloader import *
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str)
parser.add_argument("--user_id", type=str, default=None)
parser.add_argument("--window_shift", type=float, default=0.25)
parser.add_argument("--window_size", type=float, default=60)
args = parser.parse_args()

DEFAULT_SIGNAL = 'eda' # The default signal can be eda or temp as they have the minimum sampling rate in Empatica E4 device
DEVICE_MIN_SAMPLING_RATE = 4


def post_processing_ground_truth(dataset_name: str, ground_truth):
    if dataset_name == 'AffectiveROAD':
        ground_truth = np.array(ground_truth)
        # ground_truth[ground_truth == 0] = -1
        # ground_truth[ground_truth == 1] = 0
        # ground_truth[ground_truth == 2] = 1
    elif dataset_name == 'CognitiveDS':
        ground_truth = np.array(ground_truth)
        ground_truth[ground_truth < 2] = 0
        ground_truth[ground_truth > 1] = 1
    return ground_truth.tolist() 


def extract_metadata_for_user(ds_path_manager, ds_data, data_ground_truth, user_id: str): 
    # The code for this session is fairly similar to the code in the function extract_features_for_user in extract_features.py
    # The sampling rate used for metadata generation (ground-truth, groups) is the one of EDA signal and TEMP signal (if available)
    # In another word, the sampling rate of the metadata should be the minimum samplinga rate of the wearable device
    ground_truth = []
    group = []
    for task_id, signal_data in ds_data.items():
        len_signal = len(signal_data)
        step = int(args.window_shift * DEVICE_MIN_SAMPLING_RATE)
        first_iter = int(args.window_size * DEVICE_MIN_SAMPLING_RATE)
        group += [user_id for _ in range(first_iter, len_signal, step)]
        if data_ground_truth[task_id] != np.array: # If the ground-truth is a single label
            ground_truth += [data_ground_truth[task_id] for _ in range(first_iter, len_signal, step)]
        else: ground_truth += data_ground_truth[task_id].tolist() # In case the ground-truth is a list of continuous values, i.e AffectiveROAD
    ground_truth = post_processing_ground_truth(args.dataset_name, ground_truth)
    ground_truth_path = os.path.join(ds_path_manager.user_data_paths[user_id].feature_path, f'{args.window_size}_{args.window_shift}', 'ground_truth.npy')
    group_path = os.path.join(ds_path_manager.user_data_paths[user_id].feature_path, f'{args.window_size}_{args.window_shift}', 'groups.npy')
    np.save(ground_truth_path, ground_truth)
    np.save(group_path, group)
    return ground_truth, group


def extract_metadata_for_dataset(ds_path_manager, ds_data):
    # The code for this session is fairly similar to the code in the function extract_features_for_dataset in extract_features.py
    # The sampling rate used for metadata generation (ground-truth, groups) is the one of EDA signal and TEMP signal (if available)
    # In another word, the sampling rate of the metadata should be the minimum samplinga rate of the wearable device
    ground_truth = []
    groups = []
    for user_id, data in ds_data[DEFAULT_SIGNAL].items():
        gt, group = extract_metadata_for_user(ds_path_manager, data, ds_data['ground_truth'][user_id], str(user_id))
        ground_truth += gt 
        groups += group
    ground_truth_path = os.path.join(ds_path_manager.combined_stats_feature_path, f'{args.window_size}_{args.window_shift}', f'{args.dataset_name}_combined_ground_truth.npy')
    group_path = os.path.join(ds_path_manager.combined_stats_feature_path, f'{args.window_size}_{args.window_shift}', f'{args.dataset_name}_combined_groups.npy')
    np.save(ground_truth_path, ground_truth)
    np.save(group_path, groups)


if __name__ == '__main__':
    dataloader = DataLoader(args.dataset_name)
    ds_path_manager = get_datapath_manager(args.dataset_name)
    if args.user_id is None:
        ds_data = dataloader.load_dataset_data()
        extract_metadata_for_dataset(ds_path_manager, ds_data)
    else:
        ds_data = dataloader.load_user_data(args.user_id)
        _, _ = extract_metadata_for_user(ds_path_manager, ds_data[DEFAULT_SIGNAL], ds_data['ground_truth'], args.user_id)