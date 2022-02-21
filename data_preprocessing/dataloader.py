from .datapath_manager import *
import pickle
import numpy as np


class DataLoader:

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dp_manager = get_datapath_manager(dataset_name)
    

    def load_user_data(self, user_id: str):
        """
        User Data Structure:
        {
            'eda': {
                'task_id': [eda_data],
            },
            'ground_truth': {
                'task_id': [ground_truth_data],
            },
            ...
        }
        """
        user_data_paths = self.dp_manager.user_data_paths[user_id]
        user_data = pickle.load(open(user_data_paths.processed_data_path, 'rb'))
        return user_data


    def load_dataset_data(self, gen_user_data_structure: bool = False):
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
        ds_data = pickle.load(open(ds_data_path, 'rb'))
        if generate_user_data_structure:
            user_ids = ds_data['eda'].keys()
            generate_user_data_structure(self.dataset_name, user_ids)
        return ds_data

    
    def load_data_for_training(self):
        """
        Load dataset for training including combined features 
        """
        dataset = np.load(self.dp_manager.combined_feature_path)
        ground_truth = np.load(self.dp_manager.combined_ground_truth_path)
        groups = np.load(self.dp_manager.combined_groups_path)
        return dataset, ground_truth, groups
    

        