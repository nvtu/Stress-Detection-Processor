import sys
import os


data_lib = os.path.abspath('../data_processing')
signal_processing_lib = os.path.abspath('../signal_processing')
if data_lib not in sys.path:
    sys.path.append(data_lib)
if signal_processing_lib not in sys.path:
    sys.path.append(signal_processing_lib)

import argparse
from tqdm import tqdm
from typing import List
from bvp_signal_processing import BVP_Signal_Processor
from eda_signal_processing import SWT_Threshold_Denoiser, EDA_Signal_Processor
from temp_signal_processing import TEMP_Signal_Processor
from dataloader import DatasetLoader
from datapath_manager import create_folder, DataPathManager
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class StatisticalFeatureExtractor:
    def __init__(self, dataset_name: str, signal: str, window_size: float, window_shift: float):
        self.dataset_name = dataset_name
        self.signal_type = signal
        self.window_size = window_size
        self.window_shift = window_shift
        self.sampling_rate = self.__get_sampling_rate()
        self.ds_path_manager = DataPathManager(self.dataset_name)
        self.ds_data = DatasetLoader(self.dataset_name).load_dataset_data(gen_user_data_structure = True)

   
    def __get_sampling_rate(self):
        if self.signal_type == 'eda' or self.signal_type == 'temp':
            return 4
        elif self.signal_type == 'bvp':
            return 64
        return None


    def clean_signal(self, signal):
        if self.signal_type == 'eda':
            swt_denoiser = SWT_Threshold_Denoiser()
            cleaned_signal = swt_denoiser.denoise(signal)
            cleaned_signal = MinMaxScaler().fit_transform(cleaned_signal.reshape(-1, 1)).ravel()
        elif self.signal_type == 'bvp':
            bvp_processor = BVP_Signal_Processor()
            cleaned_signal = bvp_processor.clean_bvp(signal, self.sampling_rate)
            cleaned_signal = bvp_processor.min_max_norm(cleaned_signal)
        return cleaned_signal


    def extract_statistical_feature(self, signal: np.array(float)) -> np.array(float):
        features = None
        if self.signal_type == 'eda':
            eda_processor = EDA_Signal_Processor()
            features = eda_processor.eda_feature_extraction(signal, self.sampling_rate)
        elif self.signal_type == 'bvp':
            bvp_processor = BVP_Signal_Processor()
            features = bvp_processor.bvp_feature_extraction(signal, self.sampling_rate)
        elif self.signal_type == 'temp':
            temp_processor = TEMP_Signal_Processor()
            features = temp_processor.temp_feature_extraction(signal)
        return features


    def extract_features_for_user(self, user_id: str):
        data = self.ds_data[self.signal_type][user_id]
        feature_path = self.ds_path_manager.get_feature_path(user_id, self.signal, self.window_size, self.window_shift)
        features = []
        if not os.path.exists(feature_path):
            for task_id, signal_data in data.items():
                len_signal = len(signal_data)
                step = int(self.window_shift * self.sampling_rate) # The true step to slide along the time axis of the signal
                first_iter = int(self.window_size * self.sampling_rate)
                # The true index of the signal at a time-point
                for current_iter in tqdm(range(first_iter, len_signal, step)): # current_iter is "second_iter"
                    previous_iter = current_iter - first_iter
                    # Copy the signal by creating a new instance to prevent the original signal from being modified
                    signal = np.array(signal_data[previous_iter:current_iter]) 
                    # Clean signal depends on the signal type
                    signal = self.clean_signal(signal)
                    # Extract statistical features from cleaned signal
                    stats_feature = self.extract_statistical_feature(signal)  
                    features.append(stats_feature)
            np.save(feature_path, np.array(features)) 
        else:
            features = np.load(feature_path).tolist()
        return features



    def extract_features_for_dataset(self):
        stats_features = []
        for user_id, data in self.ds_data[self.signal_type].items():
            print("Processing ---- {} ----".format(user_id))
            features = self.extract_features_for_user(data, str(user_id))
            stats_features += features
            print('----------------------------------------')

        stats_features = np.array(stats_features)
        stats_features_folder_path = os.path.join(self.ds_path_manager.stats_feature_path, f'{self.window_size}_{self.window_shift}')
        create_folder(stats_features_folder_path)
        output_file_path = os.path.join(stats_features_folder_path, f'{self.signal_type}.npy')
        np.save(output_file_path, stats_features)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("signal", type=str)
    parser.add_argument("--user_id", type=str, default=None)
    parser.add_argument("--window_shift", type=float, default=0.25)
    parser.add_argument("--window_size", type=float, default=60)

    args = parser.parse_args()

    statistical_feature_extractor = StatisticalFeatureExtractor(args.dataset_name, args.signal, args.window_size, args.window_shift)

    if args.user_id is None:
        # Process statistics for all users in the dataset
        statistical_feature_extractor.extract_features_for_dataset()
    else:
        # Process statistical feature extraction for a single user
        _ = statistical_feature_extractor.extract_features_for_user(args.user_id)