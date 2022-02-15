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
    processed_dataset_path: str
    user_data_paths: Dict[str, UserDataPath]


def get_datapath_manager(dataset_name: str):
    config_file_path = str(Path(__file__).parent.parent / 'config.json')
    config = configparser.ConfigParser()
    config.read(config_file_path)

    dataset_path = config['DATA_PATH'][dataset_name]
    user_data_paths = {}
    for user_id in sorted(os.listdir(dataset_path)):
        user_data_path = UserDataPath(
            user_data_path=os.path.join(dataset_path, user_id),
            acc_path=os.path.join(dataset_path, user_id, 'ACC.csv'),
            bvp_path=os.path.join(dataset_path, user_id, 'BVP.csv'),
            eda_path=os.path.join(dataset_path, user_id, 'EDA.csv'),
            temp_path=os.path.join(dataset_path, user_id, 'TEMP.csv'),
            processed_data_path=os.path.join(dataset_path, user_id, f'{dataset_name}_{user_id}.json'),
            feature_path=os.path.join(dataset_path, user_id, 'features')
        )
        user_data_paths[user_id] = user_data_path

    ds_path_manager = DatasetPath(
        dataset_path=dataset_path,
        processed_dataset_path=os.path.join(dataset_path, f'{dataset_name}.json'),
        user_data_paths=user_data_paths
    )

    return ds_path_manager