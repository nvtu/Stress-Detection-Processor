# Add additional library
import sys, os

data_lib = os.path.abspath('../data')
eda_sp_lib = os.path.abspath('../signal_processing')
if data_lib not in sys.path:
    sys.path.append(data_lib)
if eda_sp_lib not in sys.path:
    sys.path.append(eda_sp_lib)

import numpy as np
from data_utils import *
from eda_signal_processing import *
from tqdm import tqdm
import os.path as osp
import pickle
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
warnings.filterwarnings('ignore')


EDA_SAMPLING_RATE = 4
# DATASET_NAME = 'AffectiveROAD'
DATASET_NAME = 'WESAD'
DEVICE = 'wrist'
# DEVICE =        if 'meditation' in task_id: continue
SIGNAL_NAME = 'EDA'
# NORMALIZER = '_nonorm'
NORMALIZER = ''
# NORMALIZER = '_stdnorm'

# WINDOW_SIZE = 60
WINDOW_SIZE = 120
WINDOW_SHIFT = 0.25

dp_manager = get_data_path_manager()
swt_denoiser = SWT_Threshold_Denoiser()
eda_processor = EDA_Signal_Processor()


if __name__ == '__main__':
    eda_stats_features = []

    dataset_path = osp.join(dp_manager.dataset_path, f'{DATASET_NAME}_{DEVICE}_dataset.pkl')
    dataset = pickle.load(open(dataset_path, 'rb'))

    for user_id, data in dataset['eda'].items():
        print(f"Extracting EDA Features of user {user_id}")
        for task_id, eda_signal in data.items():
            len_eda_signal = len(eda_signal)
            step = int(WINDOW_SHIFT * EDA_SAMPLING_RATE) # The true step to slide along the time axis of the signal
            first_iter = int(WINDOW_SIZE * EDA_SAMPLING_RATE) # The true index of the signal at a time-point 
            for current_iter in tqdm(range(first_iter, len_eda_signal, step)): # current_iter is "second_iter"
                previous_iter = current_iter - first_iter
                signal = swt_denoiser.denoise(eda_signal[previous_iter:current_iter])
                if NORMALIZER != '_nonorm':
                    signal = MinMaxScaler().fit_transform(signal.reshape(-1, 1)).ravel()
                    # signal = StandardScaler().fit_transform(signal.reshape(-1, 1)).ravel()
                stats_feature = eda_processor.eda_feature_extraction(signal, EDA_SAMPLING_RATE) # Extract statistical features from extracted EDA features
                eda_stats_features.append(stats_feature)


    eda_stats_features = np.array(eda_stats_features) # Transform to numpy array format
    output_file_path = osp.join(get_dataset_stats_features_path(dp_manager, DATASET_NAME), f'{DATASET_NAME}_{DEVICE}_{SIGNAL_NAME}{NORMALIZER}_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
    np.save(output_file_path, eda_stats_features)