import configparser
from typing import Dict
from dataclasses import dataclass
from pathlib import Path
import json
import os



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
    dataset_path: str
    metadata_path: str
    processed_dataset_path: str
    stats_feature_path: str
    user_data_paths: Dict[str, UserDataPath]


def get_datapath_manager(dataset_name: str):
    config_file_path = str(Path(__file__).parent.parent / 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_file_path)

    dataset_path = config['DATA_PATH'][dataset_name]
    metadata_path = os.path.join(dataset_path, 'metadata.csv')
    user_data_paths = {}
    list_users = sorted([user_id for user_id in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, user_id))])
    for user_id in list_users:
        user_data_path = UserDataPath(
            user_data_path=os.path.join(dataset_path, user_id),
            acc_path=os.path.join(dataset_path, user_id, 'ACC.csv'),
            bvp_path=os.path.join(dataset_path, user_id, 'BVP.csv'),
            eda_path=os.path.join(dataset_path, user_id, 'EDA.csv'),
            temp_path=os.path.join(dataset_path, user_id, 'TEMP.csv'),
            processed_data_path=os.path.join(dataset_path, user_id, f'{dataset_name}_{user_id}.pkl'),
            feature_path=os.path.join(dataset_path, user_id, 'features')
        )
        user_data_paths[user_id] = user_data_path

    ds_path_manager = DatasetPath(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        processed_dataset_path=os.path.join(dataset_path, f'{dataset_name}.pkl'),
        stats_feature_path=os.path.join(dataset_path, f'stats_features'),
        user_data_paths=user_data_paths
    )

    return ds_path_manager