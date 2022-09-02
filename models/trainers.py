from data_processing.dataloader import EmbeddingDataLoader
from models.models import BranchingNN, MLModel
from utils.logger import Logger
from sklearn.preprocessing import StandardScaler
from models.evaluators import Evaluator
import torch
import torch.nn as nn
import os
import numpy as np
import pickle
from typing import List


class BranchNeuralNetworkTrainer:

    """
    This is the Branching Neural Network model used in stress detection problem, which is described in the paper:
    https://arxiv.org/abs/2203.09663
    """

    def __init__(self, save_log_path: str, save_model_path: str, target_metrics: List[str], config_dict):
        super(BranchNeuralNetworkTrainer, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = BranchingNN(config_dict, self.device).to(self.device)
        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.Adam(params, lr = config_dict['learning_rate'], weight_decay = 1e-3)
        # Check if the model has training checkpoint
        if os.path.exists(save_model_path):
            print("LOAD PRETRAINED MODEL")
            model_checkpoint = torch.load(save_model_path)
            self.model.load_state_dict(model_checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])

        self.loss_func = nn.BCEWithLogitsLoss()

        self.save_log_path = save_log_path
        self.save_model_path = save_model_path # Directory where model files are saved
        self.target_metrics = target_metrics # The metrics to be used for evaluating the training process
        self.config_dict = config_dict
        self.num_branches = config_dict['num_branches']
        self.__evalutator = Evaluator(self.target_metrics)
        self.__logger = Logger(self.save_log_path)
        self.__std_scaler = StandardScaler()


    
    def __infer(self, feats):
        # Covert all feature type to the training-device datatype
        feats = feats.to(self.device)
        combined_logits, branch_logits = self.model(feats)
        return combined_logits, branch_logits

    
    def train_epoch(self, dataloader: EmbeddingDataLoader, epoch_id: int = 1, step_print_result = 10) -> List[float]:
        train_loss = [] # List of training loss for each epoch
        y_pred = []
        y_true = []

        self.model.train() # Set model to training mode

        dataloader = dataloader.make_dataloader(batch_size = self.config_dict['batch_size'], is_train = True)
        for i, (feats, labels) in enumerate(dataloader):

            combined_logits, branch_logits = self.__infer(feats)

            # Compute the loss
            labels = labels.to(self.device).unsqueeze(-1)
            branching_losses = []
            loss_combined = self.loss_func(combined_logits, labels)
            for branch_logit in branch_logits:
                branching_loss = self.loss_func(branch_logit, labels)
                branching_losses.append(branching_loss)
            loss = loss_combined + sum(branching_losses) # Final loss for the branching NN model

            # Update gradients
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Record the training metrics
            with torch.no_grad():
                sigmoid_function = nn.Sigmoid()
                combined_cls = sigmoid_function(combined_logits)
                
                train_loss.append(loss.item())
                y_pred += combined_cls.round().squeeze().cpu().numpy().tolist()
                y_true += labels.squeeze().cpu().numpy().tolist()

            # Print the training metrics
            str_info = f'Epoch: {epoch_id}, Step: {i}, Loss: {loss.item()}'
            if i % step_print_result == 0 or i == len(dataloader) - 1:
                self.__logger.append(str_info) # Log the loss
                # print(str_info) 

        # Evaluate the training metrics
        eval_results = self.__evalutator.evaluate(y_true, y_pred)
        
        # Log the epoch evaluation results
        str_info = f'----> Epoch: {epoch_id}, Loss: {np.mean(train_loss)}, Evaluation: {eval_results}'
        self.__logger.append(str_info)
        if epoch_id % step_print_result == 0 :
            print(str_info)

        eval_results = [val for _, val in eval_results.items()]
        return eval_results


    def train(self, train_dataloader: EmbeddingDataLoader, validate_dataloader: EmbeddingDataLoader = None, num_epochs: int = 5000):
        MAX_LOSS_CHANGE_ITERATION = num_epochs // 10 # Maximum number of epochs to change the learning rate
        EPS = 5e-3

        # Scale the features before training
        self.__std_scaler.fit(train_dataloader.dataset.dataset)
        train_dataloader.dataset.dataset = self.__std_scaler.transform(train_dataloader.dataset.dataset)
        if validate_dataloader is not None:
            validate_dataloader.dataset.dataset = self.__std_scaler.transform(validate_dataloader.dataset.dataset)


        cnt_loss_change = 0 # Early stopping condition
        num_metrics = len(self.target_metrics)
        optimal_cost_value = num_metrics
        for epoch_id in range(num_epochs):
            eval_metrics_results = self.train_epoch(train_dataloader, epoch_id, step_print_result = 10)
            # if validate_dataloader is not None:
            #     update_cost = num_metrics
            #     pass
            # else: update_cost = num_metrics - sum(eval_metrics_results)
            update_cost = num_metrics - sum(eval_metrics_results)

            # Update the best model if the updated metrics is smaller than the best metrics
            if optimal_cost_value > update_cost + EPS:
                optimal_cost_value = update_cost
                saved_model = {
                    'epoch': epoch_id,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(saved_model, f'{self.save_model_path}')
                # Reset the early stopping condition
                cnt_loss_change = 0
                # print(epoch_id, optimal_cost_value, epoch_id, self.predict_and_evaluate(validate_dataloader))
            else: cnt_loss_change += 1

            # Early stopping condition
            if cnt_loss_change >= MAX_LOSS_CHANGE_ITERATION:
                print('BREAK at epoch ', epoch_id)
                break
        

        model_checkpoint = torch.load(self.save_model_path)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        eval_results = self.predict_and_evaluate(validate_dataloader)
        print("TRAINING", self.predict_and_evaluate(train_dataloader))
        print('VALIDATING ', self.predict_and_evaluate(validate_dataloader))
        return eval_results

        
    def predict(self, dataloader: EmbeddingDataLoader):
        predicts = []
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            dataloader = dataloader.make_dataloader(batch_size = self.config_dict['batch_size'], is_train = False)
            for i, (feats, _) in enumerate(dataloader):
                combined_logits, _ = self.__infer(feats)
                sigmoid_function = nn.Sigmoid()
                combined_cls = sigmoid_function(combined_logits)
                predicts += combined_cls.round().squeeze().cpu().numpy().tolist()
        return predicts


    def predict_and_evaluate(self, dataloader: EmbeddingDataLoader):
        y = dataloader.dataset.ground_truth
        predicts = self.predict(dataloader)
        eval_results = self.__evalutator.evaluate(y, predicts)
        return eval_results

    
class MachineLearningModelTrainer:
    """
    The Trainer class for Machine Learning models including SVM, Random Forest, Logistic Regression, and K-Nearest Neighbors.
    NOTE: These Machine Learning models are originated from the sklearn library. Therefore, incremental training is not supported. 
    """


    def __init__(self, method: str, target_metrics: List[str], random_state: int = 0):
        self.random_state = random_state
        self.target_metrics = target_metrics
        self.method = method
        self.__std_scaler = StandardScaler()
        self.__evaluator = Evaluator(self.target_metrics)
        self.__methods_need_scaler = ['svm', 'knn', 'logistic_regression', 'sgd', 'VotingCLF']
        self.model = MLModel(method, random_state).get_classifier()


    def __fit_scaler(self, X: np.array):
        if self.method in self.__methods_need_scaler:
            self.__std_scaler.fit(X)


    def __transform_data(self, X):
        scaled_X = X
        if self.method in self.__methods_need_scaler:
            scaled_X = self.__std_scaler.transform(X)
        return scaled_X


    def train(self, train_dataloader: EmbeddingDataLoader, validate_dataloader: EmbeddingDataLoader = None, **args):
        X_train, y_train = train_dataloader.dataset.dataset, train_dataloader.dataset.ground_truth

        # Re-scaled the data so that the input is valid for some ML models training such as SVM, KNN, etc.
        self.__fit_scaler(X_train)
        X_train = self.__transform_data(X_train)

        # Train the ML model
        self.model.fit(X_train, y_train)

        # Evaluate the model with training metrics
        y_preds = self.model.predict(X_train)
        train_eval_results = self.__evaluator.evaluate(y_train, y_preds)

        # Log the train evaluation results 
        str_info = "Train Evaluation Results: {}".format(train_eval_results)
        print(str_info)

        # Validate the ML model is the validate data is available
        if validate_dataloader is not None:
            eval_results = self.predict_and_evaluate(validate_dataloader)

            # Log the validation evaluation results
            str_info = "->>> Validation Evaluation Results: {}".format(eval_results)
            print(str_info)
        else: eval_results = None

        splitter = "-----------------------------------------------------------------------------------------"
        
        return eval_results
        


    def predict(self, dataloader: EmbeddingDataLoader):
        # NOTE: In this case, the ground-truth of the dataloader is None
        X = dataloader.dataset.dataset
        X = self.__transform_data(X)

        # Predict the labels
        y_preds = self.model.predict(X)
        return y_preds


    def predict_and_evaluate(self, dataloader: EmbeddingDataLoader):
        y = dataloader.dataset.ground_truth
        y_preds = self.predict(dataloader)
        eval_results = self.__evaluator.evaluate(y, y_preds)
        return eval_results