from .dataloader import DatasetLoader
from .datapath_manager import DataPathManager
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
from typing import List, Optional
from tqdm import tqdm


class DataSplitter:
    """
    This is used to split the data for stress detection classifiers training.
    There are two conventional methods to split the data:
        1. Subject-Dependent
        2. Subject-Independent
    """

    def __init__(self, dataset_name: str, method: str):
        self.dataset_name = dataset_name
        self.dataset, self.ground_truth, self.groups \
            = DatasetLoader(dataset_name).load_data_for_training()
        self.__method = method
        self.__current_index = -1
        self.__indexed = self.__split_data()
        self.num_subjects = len(self.__indexed)


    def next(self) -> Optional[List[np.array, np.array, np.array, np.array]]:
        """
        Get next train and test user data
        """
        self.__current_index += 1
        if self.__current_index == self.num_subjects:
            return None
        return self.__get_train_test_data()
   

    def reset(self):
        self.__current_index = -1


    def __get_train_test_data(self) -> Optional[List[np.array, np.array, np.array, np.array, str]]:
        """
        This function is specially designed for next() function to get the next train & test data.
        It cannot be used solely with other function in this class.
        """
        if self.__method == 'dependent':
            # Get indices
            train_index, test_index, target_user_index = self.__indexed[self.__current_index]
            # Get targeted user's data
            user_data, user_ground_truth = self.dataset[target_user_index], self.ground_truth[target_user_index]
            # Get targeted user
            target_user = self.groups[target_user_index][0]
            # Get train and test data
            X_train, y_train, X_test, y_test = user_data[train_index], user_ground_truth[train_index], \
                user_data[test_index], user_ground_truth[test_index]
            return [X_train, y_train, X_test, y_test, target_user]
        elif self.__method == 'independent':
            # Get indices
            train_index, test_index = self.__indexed[self.__current_index]
            # Get targeted user
            target_user = self.groups[test_index][0]
            # Get train and test data
            X_train, y_train, X_test, y_test = self.dataset[train_index], self.ground_truth[train_index], \
                self.dataset[test_index], self.ground_truth[test_index]
            return [X_train, y_train, X_test, y_test, target_user]
        else: raise(ValueError("Invalid method for data splitter: {}".format(self.method)))

    
    def __split_data(self):
        if self.__method == 'dependent':
            return self.__split_data_dependent()
        elif self.__method == 'independent':
            return self.__split_data_independent()
        else: raise(ValueError('Invalid method for data splitter: {}'.format(self.method)))


    def __split_data_dependent(self, test_size: float = 0.2):

        def split_train_test(y):
            """
            Split train and test data for subject-dependent model training:
                - Train_data: (1 - test_size) * number of data of a class
                - Test_data: test_size * number of data of a class 
            NOTE: This means that this approach of data splitting simulate the real-life situation 
            where the test data is the segment of data that is recorded later after we have the train data.
            """
            train_indices = []
            test_indices = []
            for _, split_index in LeaveOneGroupOut().split(y, y=None, groups=y):
                train_index = int(len(split_index) * test_size)
                train_indices += split_index[:train_index]
                test_indices += split_index[train_index:]
            return train_indices, test_indices
        
        indices = []
        
        for _, test_index in tqdm(LeaveOneGroupOut().split(self.dataset, self.ground_truth, self.groups)):
            user_ground_truth = self.ground_truth[test_index]

            # Validate if the test set and train set have enouh classes
            if self.__validate_label_cnt(user_ground_truth) == False:
                continue

            train_indices, test_indices = split_train_test(user_ground_truth)
            indices.append((train_indices, test_indices, test_index))


    def __split_data_independent(self):
        """
        Split data for subject-independent model training:
            - Train_data: all data except a targeted subject
            - Test_data: the targeted subject
        """
        indices = []
        for train_index, test_index in tqdm(LeaveOneGroupOut().split(self.dataset, self.ground_truth, self.groups)):
            y_train, y_test = self.ground_truth[train_index], self.ground_truth[test_index]

            # Validate if the test set and train set have enouh classes
            if self.__validate_label_cnt(y_train) == False or self.__validate_label_cnt(y_test) == False:
                continue

            indices.append((train_index, test_index))
        return indices


    def __validate_label_cnt(self, y):
        num_classes = len(np.unique(y))
        return num_classes > 1


    