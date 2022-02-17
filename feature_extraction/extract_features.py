import argparse
from tqdm import tqdm
from typing import List
from ..signal_processing.bvp_signal_processing import *
from ..signal_processing.eda_signal_processing import *
from ..signal_processing.temp_signal_processing import *
from ..data_preprocessing.dataloader import DataLoader
from ..data_preprocessing.datapath_manager import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str, required=True)
parser.add_argument("--signal", type=str, required=True)
parser.add_argument("--window_shift", type=float, required=False, default=0.25)
parser.add_argument("--window_size", type=float, required=False, default=60)
parser.add_argument("--user_id", type=str, required=False, default=None)

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


if __name__ == '__main__':
    dataloader = DataLoader(args.dataset_name)
    ds_path_manager = get_datapath_manager(args.dataset_name)
    if args.user_id is None:
        # Process statistics for all users in the dataset
        stats_features = []
        sampling_rate = get_sampling_rate(args.signal)
        ds_data = dataloader.load_dataset_data()
        for user_id, data in ds_data[args.signal].items():
            print("Processing ---- {} ----".format(user_id))
            for task_id, signal_data in data.items():
                len_signal = len(signal_data)
                step = int(args.window_shift * sampling_rate) # The true step to slide along the time axis of the signal
                first_iter = int(args.window_size * sampling_rate)
                # The true index of the signal at a time-point
                for current_iter in tqdm(range(first_iter, len_signal)): # current_iter is "second_iter"
                    previous_iter = current_iter - first_iter
                    signal = signal_data[previous_iter:current_iter]
                    # Clean signal depends on the signal type
                    signal = clean_signal(signal, args.signal)
                    # Extract statistical features from cleaned signal
                    stats_feature = extract_statistical_feature(signal, args.signal)  
                    stats_features.append(stats_feature)
            print('----------------------------------------')

        stats_features = np.array(stats_features)
        output_file_path = os.path.join(ds_path_manager.stats_feature_path, f'{args.signal}.npy')
        np.save(output_file_path, stats_features)
    else:
        # Process statistical feature extraction for a single user
        pass 