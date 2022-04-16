from datapath_manager import DataPathManager
from tqdm import tqdm
import pandas as pd
import argparse
import os
import pickle



class DatasetPreprocessor:
    """
    Preprocess the raw data obtain from the Empatica E4 device 
    to generate a combined data file for feature extraction.
    NOTE: This code only works for the dataset with the data structure as defined by Empatica E4:
        dataset_name
            |- user_id
                |- acc
                |- bvp
                |- eda
                |- temp
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name 
        self.dp_manager = DataPathManager(dataset_name)


    def __load_signal_data(self, data_path: str):
        signal_data = [line.rstrip() for line in open(data_path, 'r').readlines()]
        start_time = float(signal_data[0])
        sampling_rate = float(signal_data[1])
        data = list(map(float, signal_data[2:]))
        return data, start_time, sampling_rate

        
    def __get_signal_ptr_index(self, signal_start_time: float, signal_sampling_rate: float, time_in_seconds):
        """
        Return the index of the signal pointer.
        """
        start_time_in_sconds, end_time_in_seconds = time_in_seconds
        first_pt = int((start_time_in_sconds - signal_start_time) * signal_sampling_rate)
        last_pt = int((end_time_in_seconds - signal_start_time) * signal_sampling_rate)
        return first_pt, last_pt


    def preprocess_dataset(self,):
        """
        Preprocess the whole dataset.
        The dataset structure should be like:
            dataset_name
                |- user_id
                    |- acc
                    |- bvp
                    |- eda
                    |- temp
        
        The structure of the output is:
            eda: {
                user_id: {
                    task_id: [eda_data]
                }
            },
            ...
        """

        metadata = pd.read_csv(self.dp_manager.metadata_path)

        eda = {}
        bvp = {}
        temp = {} 
        ground_truth = {}

        for user_id, user_data_path in tqdm(self.dp_manager.user_data_paths.items()):
            if not os.path.exists(user_data_path.processed_data_path):
                self.preprocess_user_data(self.dataset_name, user_id, metadata=metadata)
            user_data = pickle.load(open(user_data_path.processed_data_path, 'rb'))

            eda[user_id] = user_data['eda']
            bvp[user_id] = user_data['bvp']
            temp[user_id] = user_data['temp']
            ground_truth[user_id] = user_data['ground_truth']

        processed_dataset = {
            'eda': eda,
            'bvp': bvp,
            'temp': temp,
            'ground_truth': ground_truth
        }
        pickle.dump(processed_dataset, open(self.dp_manager.processed_dataset_path, 'wb'))


    def preprocess_user_data(self, user_id: str, metadata: pd.DataFrame = None):
        """
        Preprocess the data of a targeted user.
        The data structure should be like:
                user_id
                    |- acc
                    |- bvp
                    |- eda
                    |- temp
        
        The structure of the output is:
            eda: {
                task_id: [eda_data]
            },
            ...
        """

        # Load metadata to get the starting time of the physiological signals
        if metadata is None:
            metadata = pd.read_csv(self.dp_manager.metadata_path)
        user_metadata = metadata[metadata['user_id'] == user_id].sort_values(by='start_time_in_seconds')
        _eda, eda_start_time, EDA_SAMPLING_RATE = self.__load_signal_data(self.dp_manager.user_data_paths[user_id].eda_path)
        _bvp, bvp_start_time, BVP_SAMPLING_RATE = self.__load_signal_data(self.dp_manager.user_data_paths[user_id].bvp_path)
        _temp, temp_start_time, TEMP_SAMPLING_RATE = self.__load_signal_data(self.dp_manager.user_data_paths[user_id].temp_path)

        eda = {}
        bvp = {}
        temp = {}
        ground_truth = {}

        for value in user_metadata.values:
            _, session_id, start_time_in_seconds, end_time_in_seconds, label = value

            eda_first_pt, eda_last_pt = self.__get_signal_ptr_index(eda_start_time, EDA_SAMPLING_RATE, (start_time_in_seconds, end_time_in_seconds)) 
            eda[session_id] = _eda[eda_first_pt:eda_last_pt]

            bvp_first_pt, bvp_last_pt = self.__get_signal_ptr_index(bvp_start_time, BVP_SAMPLING_RATE, (start_time_in_seconds, end_time_in_seconds))
            bvp[session_id] = _bvp[bvp_first_pt:bvp_last_pt]

            temp_first_pt, temp_last_pt = self.__get_signal_ptr_index(temp_start_time, TEMP_SAMPLING_RATE, (start_time_in_seconds, end_time_in_seconds))
            temp[session_id] = _temp[temp_first_pt:temp_last_pt]

            ground_truth[session_id] = label # Support only one-label ground-truth

        user_data = {
            'eda': eda,
            'bvp': bvp,
            'temp': temp,
            'ground_truth': ground_truth
        } 
        pickle.dump(user_data, open(self.dp_manager.user_data_paths[user_id].processed_data_path, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, help='The name of the dataset.')
    parser.add_argument('--user_id', type=str, help='The user id in the dataset that need to be pre-processed.')
    args = parser.parse_args()
    dataset_preprocessor = DatasetPreprocessor(args.dataset_name)

    if args.user_id is None:
        dataset_preprocessor.preprocess_dataset()
    else:
        dataset_preprocessor.preprocess_user_data(args.user_id)