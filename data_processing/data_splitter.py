from dataloader import DatasetLoader
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np


class DataSplitter:
    """
    This is used to split the data for stress detection classifiers training.
    There are two conventional methods to split the data:
        1. Subject-Dependent
        2. Subject-Independent
    """

    def __init__(self, dataset_name: str, method: str, test_size: float = 1.0):
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.dataset, self.ground_truth, self.groups, self.tasks_indices \
            = DatasetLoader(dataset_name).load_data_for_training()
        self.__method = method
        self.__current_index = -1
        self.__indexed = self.__split_data()
        self.num_subjects = len(self.__indexed)

        if method == 'dependent' and self.test_size == 1.0: # The test size of the dependent method should be < 1.0
            raise(ValueError('test_size cannot be 1.0 for dependent method'))


    def next(self):
        """
        Get next train and test user data
        """
        self.__current_index += 1
        if self.__current_index == self.num_subjects:
            return None
        return self.__get_train_test_data()
   

    def reset(self):
        self.__current_index = -1

    
    def get_current_test_indices(self):
        return self.__indexed[self.__current_index] if self.__current_index >= 0 else None


    def __get_train_test_data(self):
        """
        This function is specially designed for next() function to get the next train & test data.
        It cannot be used solely with other function in this class.
        """
        if self.__method == 'dependent':
            # Get indices
            train_index, test_index, target_user_index = self.__indexed[self.__current_index]

            train_index = np.sort(train_index)
            test_index = np.sort(test_index)

            # Get targeted user's data
            # user_data, user_ground_truth = self.dataset[target_user_index], self.ground_truth[target_user_index]
            # Get targeted user
            target_user = self.groups[target_user_index][0]
            # Get train and test data
            X_train, y_train, X_test, y_test = self.dataset[train_index], self.ground_truth[train_index], \
                self.dataset[test_index], self.ground_truth[test_index]
        elif self.__method == 'independent':
            # Get indices
            train_index, test_index = self.__indexed[self.__current_index]
            # Get targeted user
            target_user = self.groups[test_index][0]
            # Get train and test data
            X_train, y_train, X_test, y_test = self.dataset[train_index], self.ground_truth[train_index], \
                self.dataset[test_index], self.ground_truth[test_index]
        else: 
            raise(ValueError("Invalid method for data splitter: {}".format(self.method)))
        return X_train, y_train, X_test, y_test, target_user

    
    def __split_data(self):
        if self.__method == 'dependent':
            return self.__split_data_dependent()
        elif self.__method == 'independent':
            return self.__split_data_independent()
        else: raise(ValueError('Invalid method for data splitter: {}'.format(self.method)))


    def split_train_test(self, indices, test_size: float = 0.3):
        """
        Split train and test data for subject-dependent model training:
            - Train_data: (1 - test_size) * number of data of a class
            - Test_data: test_size * number of data of a class 
        NOTE: This means that this approach of data splitting simulate the real-life situation 
        where the test data is the segment of data that is recorded later after we have the train data.
        """
        cut_point = int((1 - test_size) * len(indices))
        train_indices = indices[:cut_point].tolist()
        test_indices = indices[cut_point:].tolist()
        return train_indices, test_indices


    def __split_train_test_data_by_tasks(self, test_index, task_indices, test_size: float = 0.3):
        """
        Split the train/test data by tasks by 
        keeping on {test_size}% of the size of the test set 
        for each task of the targeted subject.
        """
        train_indices, test_indices = [], []
        for _, task_test_index in LeaveOneGroupOut().split(test_index, y=None, groups=task_indices):
            task_train_indices, task_test_indices = self.split_train_test(test_index[task_test_index], test_size = test_size)
            train_indices += task_train_indices
            test_indices += task_test_indices

        return train_indices, test_indices


    def __split_data_dependent(self):
        """
            Split data for subject-dependent model training:
                - Train_data: (1 - test_size) * number of data of a class 
                - Test_data: test_size * number of data of a class
        """ 
        indices = []
        
        for _, test_index in LeaveOneGroupOut().split(self.dataset, self.ground_truth, self.groups):
            user_ground_truth = self.ground_truth[test_index]
            user_task_indices = self.tasks_indices[test_index]
 
            # Validate if the test set and train set have enouh classes
            if self.__validate_label_cnt(user_ground_truth) == False:
                continue

            train_indices, test_indices = self.__split_train_test_data_by_tasks(test_index, 
                    user_task_indices, 
                    test_size = self.test_size
            )

            indices.append((train_indices, test_indices, test_index)) # test index to determine the targeted user
        return indices


    def __split_data_independent(self):
        """
        Split data for subject-independent model training:
            - Train_data: all data except a targeted subject
            - Test_data: the targeted subject
        """
        indices = []
        for train_index, test_index in LeaveOneGroupOut().split(self.dataset, self.ground_truth, self.groups):
            y_train, y_test = self.ground_truth[train_index], self.ground_truth[test_index]

            # Validate if the test set and train set have enouh classes
            if self.__validate_label_cnt(y_train) == False or self.__validate_label_cnt(y_test) == False:
                continue
                
            # NOTE: Special case for independent model, the size of the test set is the last {test_size}% 
            # of the size of the test set for each task of the targeted subject.
            if self.test_size < 1.0:
                test_task_indices = self.tasks_indices[test_index] # The task indices of the targeted test subject
                _, test_index = self.__split_train_test_data_by_tasks(test_index, 
                        test_task_indices, test_size = self.test_size)

            indices.append((train_index, test_index))
        return indices


    def __validate_label_cnt(self, y):
        num_classes = len(np.unique(y))
        return num_classes > 1