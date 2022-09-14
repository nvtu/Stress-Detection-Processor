"""
This file is created to serve the purpose of creating features for each user in the wild using Empatica E4 wrist-band device.
"""

from tqdm import tqdm
import os
import __init__
import argparse
import numpy as np
from bvp_signal_processing import BVP_Signal_Processor
from eda_signal_processing import EDA_Signal_Processor, SWT_Threshold_Denoiser
from temp_signal_processing import TEMP_Signal_Processor
from sklearn.preprocessing import MinMaxScaler
from datapath_manager import ITWDataPathManager
import warnings
warnings.filterwarnings("ignore")


class ITWStatisticalFeatureExtractor:

    def __init__(self, dataset_name: str, signal_type: str, window_size: int, window_shift: float):
        self.dataset_name = dataset_name
        self.signal_type = signal_type
        self.window_size = window_size
        self.window_shift = window_shift
        self.sampling_rate = self.__get_sampling_rate()
        pass


    def __get_sampling_rate(self):
        if self.dataset_name == 'WESAD_chest': # NOTE: Special case of WESAD_chest dataset where it is recorded using conventional clinical devices
            return 700
        else:
            if self.signal_type == 'eda' or self.signal_type == 'temp':
                return 4
            elif self.signal_type == 'bvp':
                return 64
        return None


    def clean_signal(self, signal):
        cleaned_signal = signal
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

    
    def extract_features(self, signal: np.array(float)):
        features = []
        first_iter = int(self.window_size * self.sampling_rate)
        len_signal = len(signal)
        step = int(self.window_shift * self.sampling_rate)
        index = -1
        for current_iter in tqdm(range(first_iter, len_signal, step)):
            try:
                index += 1
                previous_iter = current_iter - first_iter
                current_signal = signal[previous_iter:current_iter]
                cleaned_signal = self.clean_signal(current_signal)
                stats_feature = self.extract_statistical_feature(cleaned_signal)
                features.append(stats_feature)
            except Exception as e:
                with open('itw_logs.txt', 'a') as f:
                    print(index, file=f)
        return features
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, default="DCU_EXP2_ITW", help='Name of the dataset')
    parser.add_argument('user_id', type=str, help='User ID')
    parser.add_argument('date', type=str, help='Date of the recording')
    parser.add_argument('--signal_type', type=str, help='Signal type')
    parser.add_argument('--window_size', type=int, default=60, help='Window size in seconds')
    parser.add_argument('--window_shift', type=float, default=0.25, help='Window shift in seconds')


    args = parser.parse_args()

    assert(args.signal_type in ['bvp', 'eda', 'temp'])
    print("->>> Extracting {} features for user: {} on date {}".format(args.signal_type, args.user_id, args.date))

    # NOTE: Get the sampling rate
    itw_feature_extractor = ITWStatisticalFeatureExtractor(args.dataset_name, args.signal_type, args.window_size, args.window_shift)

    # NOTE: Create the feature extractor object

    # Get the data path
    data_path_manager = ITWDataPathManager(args.dataset_name)
    dataset_path = data_path_manager.get_dataset_path()

    # Iterate through each folder in the day of the user
    
    E4_FOLDER_NAME = 'E4'
    user_data_path = os.path.join(dataset_path, 'data', args.user_id, args.date, E4_FOLDER_NAME)
    if not os.path.exists(user_data_path):
        print("->>> Path does not exist: {}".format(user_data_path))
        exit(0)
    

    for session in os.listdir(user_data_path):
        # Create output directory if it does not exists
        output_folder_path = os.path.join(user_data_path, session, 'features')
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        print("->>> Processing session: {}".format(session))
        signal_file_path = os.path.join(user_data_path, session, args.signal_type.upper() + '.csv')
        feature_output_path = os.path.join(output_folder_path, args.signal_type.upper() + '_features.npy')

        if os.path.exists(feature_output_path):
            continue

        signal = np.array([float(line.rstrip()) for line in open(signal_file_path, 'r').readlines()][2:])

        features = itw_feature_extractor.extract_features(signal)
        np.save(feature_output_path, features)
