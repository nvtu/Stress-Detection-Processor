import os
from sqlite3 import DatabaseError
import sys
import __init__
import argparse
from datapath_manager import DataPathManager
from dataloader import DatasetLoader
from tqdm import tqdm
import numpy as np


class MetadataGenerator():
    """
    This file is used to generate and pre-process metadata for the dataset, which includes:
        1. Ground-truth
        2. Groups: The user_id label for each window-size segment data
    """

    DEFAULT_SIGNAL = 'eda' # The default signal can be eda or temp as they have the minimum sampling rate in Empatica E4 device
    DEVICE_MIN_SAMPLING_RATE = 4


    def __init__(self, dataset_name: str, window_size: int, window_shift: float, device_min_sampling_rate: int = None):
        self.dataset_name = dataset_name
        self.window_size = window_size
        self.window_shhift = window_shift
        self.device_min_sampling_rate = device_min_sampling_rate if device_min_sampling_rate is not None \
             else MetadataGenerator.DEVICE_MIN_SAMPLING_RATE
        self.ds_path_manager = DataPathManager(dataset_name).ds_path_manager
        self.ds_data = DatasetLoader(dataset_name).load_dataset_data()
        

    def post_processing_ground_truth(self, ground_truth):
        ground_truth = np.array(ground_truth)
        if self.dataset_name == 'AffectiveROAD':
            return ground_truth.tolist()
            # ground_truth[ground_truth == 0] = -1
            # ground_truth[ground_truth < 2] = 0
            # ground_truth[ground_truth == 2] = 1
        elif self.dataset_name == 'CognitiveDS':
            ground_truth[ground_truth < 1] = 0
            ground_truth[ground_truth > 0] = 1
        elif self.dataset_name == 'DCU_NVT_EXP2':
            ground_truth[ground_truth < 1] = 0
            ground_truth[ground_truth > 0] = 1
        return ground_truth.tolist() 


    def extract_metadata_for_user(self, user_id: str): 
        # The code for this session is fairly similar to the code in the function extract_features_for_user in extract_features.py
        # The sampling rate used for metadata generation (ground-truth, groups) is the one of EDA signal and TEMP signal (if available)
        # In another word, the sampling rate of the metadata should be the minimum sampling rate of the wearable device
        data_ground_truth = self.ds_data['ground_truth'][user_id]
        ground_truth = []
        group = []
        tasks_indices = []
        for task_id, signal_data in self.ds_data[MetadataGenerator.DEFAULT_SIGNAL][user_id].items():
            len_signal = len(signal_data)
            step = int(args.window_shift * self.device_min_sampling_rate)
            first_iter = int(args.window_size * self.device_min_sampling_rate)
            group += [user_id for _ in range(first_iter, len_signal, step)]

            if type(data_ground_truth[task_id]) != list: # If the ground-truth is a single label
                ground_truth += [data_ground_truth[task_id] for _ in range(first_iter, len_signal, step)]
            else: 
                # In case the ground-truth is a list of continuous values, i.e AffectiveROAD
                ground_truth += [data_ground_truth[task_id][i] for i in range(first_iter, len_signal, step)] 

            tasks_indices += [task_id for _ in range(first_iter, len_signal, step)] # The task_id is the same for all the windows in a user

        ground_truth = self.post_processing_ground_truth(ground_truth)
        ground_truth_path = os.path.join(self.ds_path_manager.user_data_paths[user_id].feature_path, f'{args.window_size}_{args.window_shift}', 'ground_truth.npy')
        group_path = os.path.join(self.ds_path_manager.user_data_paths[user_id].feature_path, f'{args.window_size}_{args.window_shift}', 'groups.npy')
        tasks_index_path = os.path.join(self.ds_path_manager.user_data_paths[user_id].feature_path, f'{args.window_size}_{args.window_shift}', 'tasks_index.npy')

        np.save(ground_truth_path, ground_truth)
        np.save(group_path, group)
        np.save(tasks_index_path, tasks_indices)
        return ground_truth, group, tasks_indices


    def extract_metadata_for_dataset(self):
        # The code for this session is fairly similar to the code in the function extract_features_for_dataset in extract_features.py
        # The sampling rate used for metadata generation (ground-truth, groups) is the one of EDA signal and TEMP signal (if available)
        # In another word, the sampling rate of the metadata should be the minimum samplinga rate of the wearable device
        ground_truth = []
        groups = []
        tasks_indices = []
        for user_id, data in self.ds_data[MetadataGenerator.DEFAULT_SIGNAL].items():
            gt, group, user_tasks_indices = self.extract_metadata_for_user(str(user_id))
            ground_truth += gt 
            groups += group
            tasks_indices += user_tasks_indices
        ground_truth_path = os.path.join(self.ds_path_manager.combined_stats_feature_path, f'{args.window_size}_{args.window_shift}', f'{args.dataset_name}_combined_ground_truth.npy')
        group_path = os.path.join(self.ds_path_manager.combined_stats_feature_path, f'{args.window_size}_{args.window_shift}', f'{args.dataset_name}_combined_groups.npy')
        tasks_index_path = os.path.join(self.ds_path_manager.combined_stats_feature_path, f'{args.window_size}_{args.window_shift}', f'{args.dataset_name}_combined_tasks_index.npy')

        np.save(ground_truth_path, ground_truth)
        np.save(group_path, groups)
        np.save(tasks_index_path, tasks_indices)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--user_id", type=str, default=None)
    parser.add_argument("--window_shift", type=float, default=0.25)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument('--device_min_sampling_rate', type=int, default=4, 
        help="Min sampling rate of the recording device")

    args = parser.parse_args()

    metadata_generator = MetadataGenerator(args.dataset_name, args.window_size, args.window_shift, 
        device_min_sampling_rate = args.device_min_sampling_rate)
    if args.user_id is None:
        metadata_generator.extract_metadata_for_dataset()
    else:
        _, _ = metadata_generator.extract_metadata_for_user(args.user_id)