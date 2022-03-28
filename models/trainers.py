from models.evaluators import Evaluator
import torch
import torch.nn as nn
from typing import List
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainBranchingNN():


    def __init__(self, model, optimizer, loss_func, save_log_dir, save_model_dir, target_metrics: List[str], config_dict):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_func = loss_func.to(device) if loss_func is not None else loss_func
        self.save_log_dir = save_log_dir # Directory where log.txt files are saved
        self.save_model_dir = save_model_dir # Directory where model files are saved
        self.target_metrics = target_metrics # The metrics to be used for evaluating the training process
        self.config_dict = config_dict
        self.num_branches = config_dict['num_branches']

    
    def train_epoch(self, dataloader, epoch_id: int = 1, step_print_result = 100):
        train_loss = [] # List of training loss for each epoch
        y_pred = []
        y_true = []

        self.model.train() # Set model to training mode
        for i, (feats, labels) in enumerate(dataloader):
            # Covert all feature type to the training-device datatype
            feats = feats.to(device)
            labels = labels.to(device).unsqueeze(-1)

            # Pass to the model 
            combined_logits, branch_logits = self.model(feats)

            # Compute the loss
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
                y_true += labels

            # Print the training metrics
            str_info = f'Epoch: {epoch_id}, Step: {i}, Loss: {loss.item()}'
            if i % step_print_result == 0 or i == len(dataloader)-1:
                print(str_info) 
        
        eval_results = Evaluator(self.target_metrics).evaluate(y_true, y_pred)
        return eval_results


    def train(self, train_dataloader, validate_dataloader = None, num_epochs: int = 100):
        EPS = 5e-3
        cnt_loss_change = 0
        best_metric = len(self.target_metrics)
        for epoch_id in range(num_epochs):
            eval_metrics_results = self.train_epoch(train_dataloader, epoch_id)
