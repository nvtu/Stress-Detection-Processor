from functools import total_ordering
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# ACKNOWLEDGEMENT
# This code is referred from the github repository 
# https://github.com/m2man/Stress-Detection-with-FC developed by 
# my colleague -- Manh-Duy Nguyen

# ----------------- Implementation of Swish activation function -----------------
class SwishImplementation(torch.autograd.Function):
    

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
    

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):


    def forward(self, x):
        return SwishImplementation.apply(x)

# --------------------------------------------------------------------------------


def activation_function(name: str = 'relu'):
    # relu , swish, leaky relu, gelu
    # Return ReLU activation function by default
    if name == 'swish':
        return MemoryEfficientSwish()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(0.2)
    elif name == 'gelu':
        return nn.GELU()
    return nn.ReLU()


class MLP(nn.Module):
    """
    A Simple Neural Network (Multilayer-Perceptron)
    """

    def __init__(self, input_size, dropout=0.4, embedding_size=[64], activation='relu'):
        super(MLP, self).__init__()
        modules = []
        modules.append(nn.Linear(input_size, embedding_size[0])) # Append the first layer
        num_embeddings = len(embedding_size)
        for idx in range(1, num_embeddings):
            modules.append(nn.Linear(embedding_size[idx-1], embedding_size[idx]))
            # Append the additional feature processing layers if it is not the last layer
            if idx != len(embedding_size) - 1: 
                modules.append(nn.BatchNorm1d(num_features=embedding_size[idx]))
                modules.append(activation_function(name = activation))
                if dropout is not None:
                    modules.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*modules)
            

    def forward(self, x):
        return self.mlp(x)


class BranchingNN(nn.Module):
    """
    A Branching Neural Network consists of different MLPs for different branches
    and then merge them together to form a final prediction
    """


    def __init__(self, config_dict):
        super(BranchingNN, self).__init__()

        dropout = config_dict['dropout']
        self.feature_dims = config_dict['feature_dims']
        activation_func = config_dict['activation_function']
        self.num_branches = config_dict['num_branches']

        assert(type(self.feature_dims) == list) # Assert that the feature_dims is a list
        assert(len(self.feature_dims) == self.num_branches) # Assert that the number of feature dimensions is the same as the number of branches

        self.branches = []
        self.logits = []
        for i in range(self.num_branches):
            embedded_feature = MLP(input_size=self.feature_dims[i], dropout=dropout, embedding_size=[self.feature_dims[i] * 2, self.feature_dims[i]], activation=activation_func) # Initialize the MLP for each branch
            logit = MLP(input_size=self.feature_dims[i], dropout=dropout, embedding_size=[1], activation=activation_func)
            self.branches.append(embedded_feature)
            self.logits.append(logit)
        
        total_feature_dims = np.sum(self.feature_dims)
        self.combined_nn = MLP(input_size=total_feature_dims, dropout=dropout, embedding_size=[total_feature_dims, 1], activation=activation_func)

    
    def forward(self, x):
        """
        Forward pass of the BranchingNN
        x: feature dimensions should be the sum of all self.feature_dims
        """
        embedding_features = []
        logits_cls = []
        prev_index = 0
        for i in range(self.num_branches):
            feat = x[prev_index:self.feature_dims[i]]
            embed = self.branches[i](feat)
            cls = self.logits[i](embed)
            logits_cls.append(cls)
            embedding_features.append(embed)
            prev_index = self.feature_dims[i]
        combined_embedding_features = torch.cat(embedding_features, axis=-1)
        combined_logit = self.combined_nn(combined_embedding_features)
        return combined_logit, logits_cls
