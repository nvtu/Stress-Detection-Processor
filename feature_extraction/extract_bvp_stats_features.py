# Add additional library
import sys, os

data_lib = os.path.abspath('../data')
bvp_sp_lib = os.path.abspath('../signal_processing')
if data_lib not in sys.path:
    sys.path.append(data_lib)
if bvp_sp_lib not in sys.path:
    sys.path.append(bvp_sp_lib)

import numpy as np
from data_utils import *
from bvp_signal_processing import *
from tqdm import tqdm
import os.path as osp
import pickle
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')


BVP_SAMPLING_RATE = 64
# DATASET_NAME = 'AffectiveROAD'
DATASET_NAME = 'WESAD'
DEVICE = 'wrist'
# DEVICE = 'right'
SIGNAL_NAME = 'BVP'
# NORMALIZER = '_stdnorm'
# NORMALIZER = '_nonorm'
NORMALIZER = ''

# WINDOW_SIZE = 60
WINDOW_SIZE = 120
WINDOW_SHIFT = 0.25

dp_manager = get_data_path_manager()
bvp_processor = BVP_Signal_Processor()

if __name__ == '__main__':
    dataset_path = osp.join(dp_manager.dataset_path, f'{DATASET_NAME}_{DEVICE}_dataset.pkl')
    dataset = pickle.load(open(dataset_path, 'rb'))

    bvp_stats_features = []
    for user_id, data in dataset['bvp'].items():
        print(f"Extracting BVP Features of user {user_id}")
        for task_id, bvp_signal in data.items():
            len_bvp_signal = len(bvp_signal)
            step = int(WINDOW_SHIFT * BVP_SAMPLING_RATE) # The true step to slide along the time axis of the signal
            first_iter = int(WINDOW_SIZE * BVP_SAMPLING_RATE) # The true index of the signal at a time-point 
            for current_iter in tqdm(range(first_iter, len_bvp_signal, step)): # current_iter is "second_iter"
                previous_iter = current_iter - first_iter
                signal = bvp_signal[previous_iter:current_iter]
                cleaned_signal = bvp_processor.clean_bvp(signal, BVP_SAMPLING_RATE)
                if NORMALIZER != '_nonorm':
                    cleaned_signal = bvp_processor.min_max_norm(cleaned_signal)
                    # cleaned_signal = bvp_processor.standard_norm(cleaned_signal)
                stats_feature = bvp_processor.bvp_feature_extraction(cleaned_signal, BVP_SAMPLING_RATE) # Extract statistical features from extracted EDA features
                bvp_stats_features.append(stats_feature)

    bvp_stats_features = np.array(bvp_stats_features) # Transform to numpy array format
    output_file_path = osp.join(get_dataset_stats_features_path(dp_manager, DATASET_NAME), f'{DATASET_NAME}_{DEVICE}_{SIGNAL_NAME}{NORMALIZER}_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
    np.save(output_file_path, bvp_stats_features)