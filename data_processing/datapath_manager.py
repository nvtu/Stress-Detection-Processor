import configparser
from tkinter import W
from typing import Dict
from dataclasses import dataclass
from pathlib import Path
import os
from typing import List
from functools import lru_cache


def create_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


@dataclass
class UserDataPath:

    user_data_path: str
    acc_path: str
    bvp_path: str
    eda_path: str
    temp_path: str
    processed_data_path: str
    feature_path: str # Output feature path


@dataclass
class DatasetPath:
    """
    Structure of a dataset folder:
        dataset_name
            |__ user_data_path
            |       |_____ user_id
            |__ stats_features: Combined statistical features
            |__ processed_dataset_path.pkl: Dataset after preprocessing
            |__ metadata.csv: Metadata of the dataset containing the user_id, session_id, start_time, end_time, label
    """
    dataset_path: str
    metadata_path: str
    processed_dataset_path: str
    stats_feature_path: str
    user_data_paths: Dict[str, UserDataPath]

   
class DataPathManager:

    """
    Data Path Manager: Get all the neccesary path of the dataset
    """
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.ds_path_manager = self.__get_datapath_manager()

   

    def get_feature_path(self, user_id: str, signal_type: str, window_size: float, window_shift: float):
        """
        Get statistical feature path of a user
        The structure of the statistical feature path is as follows:
            dataset_path
                |__ data
                    |__ user_id
                        |__ features
                            |__ {window_size}_{window_shift}
                                |__ {signal_type}.npy
        """
        folder_path = os.path.join(self.ds_path_manager.user_data_paths[user_id].feature_path, f'{window_size}_{window_shift}')
        create_folder(folder_path)
        feature_path = os.path.join(folder_path, f'{signal_type}.npy')
        if not os.path.exists(feature_path):
            raise ValueError(f'Signal type {signal_type} with {window_size} -- {window_shift} has not yet been created!')
        return feature_path
        

    @lru_cache(maxsize=None)
    def __get_datapath_manager(self):
        """
        Create Datapath Manager from a dataset
        """

        config_file_path = str(Path(__file__).parent.parent / 'config.ini')
        config = configparser.ConfigParser()
        config.read(config_file_path)

        dataset_path = config['DATA_PATH'][self.dataset_name]
        metadata_path = os.path.join(dataset_path, 'metadata.csv')

        # Construct user data paths
        user_data_paths = {}
        data_path = os.path.join(dataset_path, 'data')
        list_users = sorted([user_id for user_id in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, user_id))])
        for user_id in list_users:
            stats_feature_folder = os.path.join(data_path, user_id, 'features')
            create_folder(stats_feature_folder)
            user_data_path = UserDataPath(
                user_data_path = os.path.join(data_path, user_id),
                acc_path = os.path.join(data_path, user_id, 'ACC.csv'),
                bvp_path = os.path.join(data_path, user_id, 'BVP.csv'),
                eda_path = os.path.join(data_path, user_id, 'EDA.csv'),
                temp_path = os.path.join(data_path, user_id, 'TEMP.csv'),
                processed_data_path = os.path.join(data_path, user_id, f'{self.dataset_name}_{user_id}.pkl'),
                feature_path = stats_feature_folder,
            )
            user_data_paths[user_id] = user_data_path

        ds_path_manager = DatasetPath(
            dataset_path = dataset_path,
            metadata_path = metadata_path,
            processed_dataset_path = os.path.join(dataset_path, f'{self.dataset_name}.pkl'),
            stats_feature_path = os.path.join(dataset_path, f'stats_features'),
            user_data_paths = user_data_paths
        )

        return ds_path_manager




