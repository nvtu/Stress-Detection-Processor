from datapath_manager import *
import json


class DataLoader:

    def __init__(self, dataset_name: str):
        self.dp_manager = get_datapath_manager(dataset_name)
    

    def load_user_data(self, user_id: str):
        """
        User Data Structure:
        {
            'task_id': {
                'eda': [eda_data],
                'bvp': [bvp_data],
                'temp': [temp_data],
                'ground_truth': ground_truth_data
            }
        }
        """
        user_data_paths = self.dp_manager.user_data_paths[user_id]
        user_data = json.load(open(user_data_paths.processed_data_path, 'r'))
        return user_data


    def load_dataset_data(self):
        """
        Dataset Structure:
        {
            'eda': {
                'user_id': {
                    'task_id': [eda_data]
                }
            },
            'bvp': {
                'user_id': {
                    'task_id': [bvp_data]
                }
            },
            ...
        }
        """
        ds_data_path = self.dp_manager.processed_dataset_path
        ds_data = json.load(open(ds_data_path, 'r'))
        return ds_data
