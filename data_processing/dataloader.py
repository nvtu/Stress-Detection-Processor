from datapath_manager import *
import pickle
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch


class DatasetLoader:

    """
    Data Loader Management
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dp_manager = DataPathManager(self.dataset_name).ds_path_manager


    def __generate_user_data_structure(self, user_ids: List[str]):
        """
        In case the dataset was pre-processed, generate the user data structure on the machine for caching process
        """

        data_path = os.path.join(self.dp_manager.dataset_path, 'data')
        for user_id in user_ids:
            user_folder_path = os.path.join(data_path, str(user_id))
            create_folder(user_folder_path)
            feature_folder_path = os.path.join(user_folder_path, 'features')
            create_folder(feature_folder_path)

    

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
        DEFAULT_FIELD = 'eda' # Default field to get the user ids in the dataset

        ds_data_path = self.dp_manager.processed_dataset_path
        ds_data = pickle.load(open(ds_data_path, 'rb'))
        if gen_user_data_structure:
            user_ids = ds_data[DEFAULT_FIELD].keys()
            self.__generate_user_data_structure(user_ids)
        return ds_data

    
    def load_data_for_training(self, window_shift: float = 0.25, window_size: int = 60):
        """
        Load dataset for training including combined features 
        """
        combined_feature_folder = os.path.join(self.dp_manager.combined_stats_feature_path, f'{window_size}_{window_shift}')
        combined_stats_feature_path = os.path.join(combined_feature_folder, f'{self.dataset_name}_combined_features.npy')
        combined_group_path = os.path.join(combined_feature_folder, f'{self.dataset_name}_combined_groups.npy')
        combined_ground_truth_path = os.path.join(combined_feature_folder, f'{self.dataset_name}_combined_ground_truth.npy')
        combined_tasks_indices_path = os.path.join(combined_feature_folder, f'{self.dataset_name}_combined_tasks_index.npy')

        dataset = np.load(combined_stats_feature_path)
        groups = np.load(combined_group_path)
        ground_truth = np.load(combined_ground_truth_path)
        tasks_indices = np.load(combined_tasks_indices_path)
        return dataset, ground_truth, groups, tasks_indices


class EmbeddingDataSet(Dataset):

    def __init__(self, dataset, ground_truth):
        """
            - dataset: a numpy array of shape (num_samples, num_features)
            - ground_truth: a numpy array of shape (num_samples, )
        """
        self.dataset = dataset
        self.ground_truth = ground_truth
    

    def __getitem__(self, index):
        sample = self.dataset[index, :]
        label = self.ground_truth[index]
        return sample, label


    def __len__(self):
        return len(self.ground_truth)


class EmbeddingDataLoader:


    def __init__(self, dataset, ground_truth):
        self.dataset = EmbeddingDataSet(dataset, ground_truth)
        

    def generate_batch(self, batch):
        feats, labels = zip(*batch)
        feats = torch.tensor(np.array([x for x in feats]))
        labels = torch.tensor(labels).float()
        return feats, labels


    def make_dataloader(self, is_train = True, **args):
        if is_train:
            num_classes = Counter(np.array(self.dataset.ground_truth).astype(int))
            classes = np.array([num_classes[key] for key in sorted(num_classes.keys())])
            weight = 1. / classes
            sample_weights = np.array([weight[cls] for cls in self.dataset.ground_truth])
            sample_weights = torch.from_numpy(sample_weights)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        else:
            sampler = None
        data = DataLoader(self.dataset, collate_fn=self.generate_batch, sampler=sampler, **args)
        return data