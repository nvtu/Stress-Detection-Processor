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
from bvp_signal_processing import *
from eda_signal_processing import *
from temp_signal_processing import *
from dataloader import DataLoader
from datapath_manager import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str)
parser.add_argument("signal", type=str)
parser.add_argument("--window_shift", type=float, default=0.25)
parser.add_argument("--window_size", type=float, default=60)
parser.add_argument("--user_id", type=str, default=None)

args = parser.parse_args()


def get_sampling_rate(signal_type: str):
    if signal_type == 'eda' or signal_type == 'temp':
        return 4
    elif signal_type == 'bvp':
        return 64
    return None


def clean_signal(signal, signal_type: str):
    cleaned_signal = signal.copy()
    sampling_rate = get_sampling_rate(signal_type)
    if signal_type == 'eda':
        swt_denoiser = SWT_Threshold_Denoiser()
        cleaned_signal = swt_denoiser.denoise(cleaned_signal)
        cleaned_signal = MinMaxScaler().fit_transform(cleaned_signal.reshape(-1, 1)).ravel()
    elif signal_type == 'bvp':
        bvp_processor = BVP_Signal_Processor()
        cleaned_signal = bvp_processor.clean_bvp(cleaned_signal, sampling_rate)
        cleaned_signal = bvp_processor.min_max_norm(cleaned_signal)
    return cleaned_signal


def extract_statistical_feature(signal, signal_type: str):
    features = None
    sampling_rate = get_sampling_rate(signal_type)
    if signal_type == 'eda':
        eda_processor = EDA_Signal_Processor()
        features = eda_processor.eda_feature_extraction(signal, sampling_rate)
    elif signal_type == 'bvp':
        bvp_processor = BVP_Signal_Processor()
        features = bvp_processor.bvp_feature_extraction(signal, sampling_rate)
    elif signal_type == 'temp':
        temp_processor = TEMP_Signal_Processor()
        features = temp_processor.temp_feature_extraction(signal)
    return features


def extract_features_for_user(ds_path_manager, data, user_id: str):
    sampling_rate = get_sampling_rate(args.signal)
    feature_path = get_feature_path(ds_path_manager, user_id, args.signal, args.window_size, args.window_shift)
    features = []
    if not os.path.exists(feature_path):
        for task_id, signal_data in data.items():
            len_signal = len(signal_data)
            step = int(args.window_shift * sampling_rate) # The true step to slide along the time axis of the signal
            first_iter = int(args.window_size * sampling_rate)
            # The true index of the signal at a time-point
            for current_iter in tqdm(range(first_iter, len_signal, step)): # current_iter is "second_iter"
                previous_iter = current_iter - first_iter
                signal = np.array(signal_data[previous_iter:current_iter])
                # Clean signal depends on the signal type
                signal = clean_signal(signal, args.signal)
                # Extract statistical features from cleaned signal
                stats_feature = extract_statistical_feature(signal, args.signal)  
                features.append(stats_feature)
        np.save(feature_path, np.array(features)) 
    else:
        features = np.load(feature_path).tolist()
    return features



def extract_features_for_dataset(ds_path_manager, ds_data):
    stats_features = []
    sampling_rate = get_sampling_rate(args.signal)
    for user_id, data in ds_data[args.signal].items():
        print("Processing ---- {} ----".format(user_id))
        features = extract_features_for_user(ds_path_manager, data, str(user_id))
        stats_features += features
        print('----------------------------------------')

    stats_features = np.array(stats_features)
    stats_features_folder_path = os.path.join(ds_path_manager.stats_feature_path, f'{args.window_size}_{args.window_shift}')
    create_folder(stats_features_folder_path)
    output_file_path = os.path.join(stats_features_folder_path, f'{args.signal}.npy')
    np.save(output_file_path, stats_features)


if __name__ == '__main__':
    dataloader = DataLoader(args.dataset_name)
    ds_data = dataloader.load_dataset_data(gen_user_data_structure = True)
    ds_path_manager = get_datapath_manager(args.dataset_name)
    if args.user_id is None:
        # Process statistics for all users in the dataset
        extract_features_for_dataset(ds_path_manager, ds_data)
    else:
        # Process statistical feature extraction for a single user
        _ = extract_features_for_user(ds_path_manager, ds_data[args.signal][args.user_id], args.user_id)