import numpy as np
from data_processing.dataloader import EmbeddingDataLoader
from data_processing.datapath_manager import DataPathManager
from models.trainers import BranchNeuralNetworkTrainer, MachineLearningModelTrainer
from tqdm import tqdm
from typing import List, Dict
from data_processing.data_splitter import DataSplitter
from utils.logger import Logger


class BinaryStressClassifier:

    def __init__(self, dataset_name: str, strategy: str,
                    model_type : str,
                    random_state: int = 0,
                    window_shift: float = 0.25,
                    window_size: int = 60,
                    target_metrics: List[str] = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']):
        self.dataset_name = dataset_name
        self.strategy = strategy # Stress detection strategy - possible options: mlp, knn, svm, logistic_regression, random_forest
        self.target_metrics = target_metrics
        self.random_state = random_state
        self.model_type = model_type.lower() # Possible options: subject_dependent, subject_independent
        self.window_shift = window_shift
        self.window_size = window_size
    

    def train(self):
        str_info = "---- Training {} {} model of {} dataset with window size = {} and window shift = {}... ----"\
            .format(self.model_type, self.strategy, self.dataset_name, self.window_size, self.window_shift)
        print(str_info)

        ds_path_manager = DataPathManager(self.dataset_name)
        saved_log_path = ds_path_manager.get_log_path(self.strategy, self.model_type, self.window_size, self.window_shift)
        
        logger = Logger(saved_log_path)
        logger.write(str_info)

        # Data splitter has already taken the responsibility to split the data according to the dependent/independent method
        ds_splitter = DataSplitter(self.dataset_name, self.model_type) 
        for _ in tqdm(range(ds_splitter.num_subjects)):
            # Get next user data for training
            data = ds_splitter.next()

            X_train, y_train, X_test, y_test, target_user = data
            str_info = "Evaluating the model for user: {}".format(target_user)
            print(str_info)
            logger.append(str_info)

            # Create the EmbeddingDataLoader object for trainer input
            train_embedding_dl = EmbeddingDataLoader(X_train, y_train)
            validate_embedding_dl = EmbeddingDataLoader(X_test, y_test)

            # Generate the directories to save models and log results
            saved_model_path = ds_path_manager.get_saved_model_path(target_user, self.strategy, self.model_type, self.window_size, self.window_shift)

            if self.strategy in ['knn', 'random_forest', 'svm']:

                # Create the model for training
                model = MachineLearningModelTrainer(self.strategy, saved_log_path, saved_model_path, self.target_metrics,
                    self.random_state)

                # Train the model
                model.train(train_embedding_dl, validate_embedding_dl)
            elif self.strategy in ['branch_neural_network']:
                model = BranchNeuralNetworkTrainer(self.strategy, self.random_state, self.window_shift, self.window_size)