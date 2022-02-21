import configparser
from typing import Dict
from dataclasses import dataclass
from pathlib import Path
import json
import os
from typing import List


@dataclass
class UserDataPath:

    user_data_path: str
    acc_path: str
    bvp_path: str
    eda_path: str
    temp_path: str
    processed_data_path: str
    feature_path: str # Output feature path
    acc_feature_path: str
    bvp_feature_path: str
    eda_feature_path: str
    temp_feature_path: str
    ground_truth_path: str
    group_path: str


@dataclass
class DatasetPath:
    dataset_path: str
    metadata_path: str
    processed_dataset_path: str
    combined_feature_path: str
    combined_ground_truth_path: str
    combined_groups_path: str
    stats_feature_path: str
    user_data_paths: Dict[str, UserDataPath]


def create_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    

def get_feature_path(ds_path_manager: DatasetPath, user_id: str, signal_type: str):
    if signal_type == 'acc':
        return ds_path_manager.user_data_paths[user_id].acc_feature_path
    elif signal_type == 'bvp':
        return ds_path_manager.user_data_paths[user_id].bvp_feature_path
    elif signal_type == 'eda':
        return ds_path_manager.user_data_paths[user_id].eda_feature_path
    elif signal_type == 'temp':
        return ds_path_manager.user_data_paths[user_id].temp_feature_path
    else:
        raise ValueError(f'Signal type {signal_type} is not supported')
    

def generate_user_data_structure(dataset_name: str, user_ids: List[str]):
    dp_manager = get_datapath_manager(dataset_name)
    data_path = os.path.join(dp_manager.dataset_path, 'data')
    create_folder(data_path)
    for user_id in user_ids:
        user_folder_path = os.path.join(data_path, str(user_id))
        create_folder(user_folder_path)
        feature_folder_path = os.path.join(user_folder_path, 'features')
        create_folder(feature_folder_path)


def get_datapath_manager(dataset_name: str):
    config_file_path = str(Path(__file__).parent.parent / 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_file_path)

    dataset_path = config['DATA_PATH'][dataset_name]
    metadata_path = os.path.join(dataset_path, 'metadata.csv')
    user_data_paths = {}
    data_path = os.path.join(dataset_path, 'data')
    list_users = sorted([user_id for user_id in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, user_id))])
    for user_id in list_users:
        stats_feature_folder = os.path.join(data_path, user_id, 'features')
        create_folder(stats_feature_folder)
        user_data_path = UserDataPath(
            user_data_path=os.path.join(data_path, user_id),
            acc_path=os.path.join(data_path, user_id, 'ACC.csv'),
            bvp_path=os.path.join(data_path, user_id, 'BVP.csv'),
            eda_path=os.path.join(data_path, user_id, 'EDA.csv'),
            temp_path=os.path.join(data_path, user_id, 'TEMP.csv'),
            processed_data_path=os.path.join(data_path, user_id, f'{dataset_name}_{user_id}.pkl'),
            feature_path=stats_feature_folder,
            acc_feature_path=os.path.join(stats_feature_folder, 'acc.npy'),
            bvp_feature_path=os.path.join(stats_feature_folder, 'bvp.npy'),
            eda_feature_path=os.path.join(stats_feature_folder, 'eda.npy'),
            temp_feature_path=os.path.join(stats_feature_folder, 'temp.npy'),
            ground_truth_path=os.path.join(stats_feature_folder, 'ground_truth.npy'),
            group_path=os.path.join(stats_feature_folder, 'group.npy'),
        )
        user_data_paths[user_id] = user_data_path

    ds_path_manager = DatasetPath(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        processed_dataset_path=os.path.join(dataset_path, f'{dataset_name}.pkl'),
        combined_feature_path=os.path.join(dataset_path, f'{dataset_name}_combined_features.npy'),
        combined_ground_truth_path = os.path.join(dataset_path, f'{dataset_name}_combined_ground_truth.npy'),
        combined_groups_path=os.path.join(dataset_path, f'{dataset_name}_combined_groups.npy'),
        stats_feature_path=os.path.join(dataset_path, f'stats_features'),
        user_data_paths=user_data_paths
    )

    return ds_path_manager
