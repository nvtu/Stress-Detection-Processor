import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_validate, GridSearchCV, train_test_split, ParameterGrid
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from collections import Counter
from tqdm import tqdm


class BinaryClassifier:

    def __init__(self, X: np.array, y: np.array, strategy: str, groups: np.array = None,
                    random_state: int = 0, 
                    subject_dependent: bool = False, 
                    subject_independent: bool = False,
                    hybrid_evaluation: bool = False):
        self.X = X
        self.y = y
        self.strategy = strategy # Stress detection strategy - possible options: mlp, knn, svm, logistic_regression, random_forest
        self.groups = groups
        self.random_state = random_state
        self.subject_dependent = subject_dependent # Subject-dependent model evaluation based on cross-validation approach
        self.subject_independent = subject_independent # Subject-independent model evaluation based on Leave one group out approach
        self.hybrid_evaluation = hybrid_evaluation
    

    def __get_hyper_parameters(self, method): # Deprecated
        params = dict()
        if method == 'random_forest':
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 500, stop = 1000, num = 2)]
            # Number of features to consider at every split
            # Minimum number of samples required to split a node
            min_samples_split = [2, 4]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 4]
            max_depth = [4, 8]
            # Class Weights
            class_weight = [None, 'balanced']
            # Method of selecting samples for training each tree
            # Create the random grid
            params = {'n_estimators': n_estimators,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'max_depth': max_depth,
                        'class_weight': class_weight,
            }
        elif method == 'logistic_regression':
            params = {'C': [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 10], 'class_weight': ['balanced', None] }
        elif method == 'svm':
            params = {'C': [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 10], 'class_weight': ['balanced', None] }
        elif method == 'mlp':
            params = { 'hidden_layer_sizes': [(64,), (128,), (256,), (512,)] } 
        elif method == 'knn':
            params = { 'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'] }
        return params
    

    def __get_classifier(self, method):
        clf = None 
        if method == 'random_forest':
            clf = RandomForestClassifier(n_estimators = 250, random_state = self.random_state, n_jobs = -1, max_features='sqrt', max_depth=8, min_samples_split=2, min_samples_leaf=4,
                                oob_score=True, bootstrap=True, class_weight = 'balanced', )
        elif method == 'logistic_regression':
            clf = LogisticRegression(random_state = self.random_state)
        elif method == 'svm':
            clf = SVC(C = 10, random_state = self.random_state, class_weight = 'balanced')
        elif method == 'mlp':
            clf = MLPClassifier(random_state = self.random_state, early_stopping = True, max_iter = 1000, activation = 'logistic')
        elif method == 'knn':
            clf = KNeighborsClassifier(n_jobs = -1, weights = 'distance')
        elif method == 'Voting3CLF':
            estimators = [('rf', self.__get_classifier('random_forest')), 
                ('svm', self.__get_classifier('svm')), 
                ('mlp', self.__get_classifier('mlp'))]
            clf = VotingClassifier(estimators = estimators, n_jobs = -1, verbose = True)
        return clf


    def __transform_data(self, method, X_train, X_test): # Transform the data using Standard Scaler
        scaled_X_train = X_train
        scaled_X_test = X_test
        if method in ['mlp', 'svm', 'knn', 'Voting3CLF']: # Only use for MLP, SVM, and KNN as these methods are sensitive to feature scaling
            std_scaler = StandardScaler()
            std_scaler.fit(X_train)
            scaled_X_train = std_scaler.transform(X_train)
            scaled_X_test = std_scaler.transform(X_test)
        return scaled_X_train, scaled_X_test


    def subject_dependent_evaluate(self, method: str):
        
        def split_train_test(X, y, test_size = 0.2):
            X_train = []
            y_train = []
            X_test = []
            y_test = []
            logo = LeaveOneGroupOut()
            for _, split_index in logo.split(X, y=None, groups=y):
                train_index = int(len(y[split_index]) * test_size)
                X_train += X[split_index][:train_index].tolist()
                y_train += y[split_index][:train_index].tolist()
                X_test += X[split_index][train_index:].tolist()
                y_test += y[split_index][train_index:].tolist()
            return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

        logo = LeaveOneGroupOut()
        results = []
        for _, test_index in tqdm(logo.split(self.X, self.y, self.groups)):
            user_dataset, user_ground_truth = self.X[test_index], self.y[test_index]
            train_subject = self.groups[test_index][0]        
            # Validate if the test set and train set have enough classes
            if self.__validate_label_cnt(user_ground_truth) == False:
                continue
            
            # Split the dataset into train and test
            X_train, y_train, X_test, y_test = split_train_test(user_dataset, user_ground_truth)
            # print(np.unique(y_train), np.unique(y_test))
            _X_train, _X_test = self.__transform_data(method, X_train, X_test)
            # Train
            clf = self.__get_classifier(method)
            clf.fit(_X_train, y_train)
            # Infer on train and test set for evaluation
            y_preds = clf.predict(_X_test)
            y_preds_train = clf.predict(_X_train)
            # Evaluate
            train_scores = self.evaluate(y_train, y_preds_train)
            train_acc, train_ba, train_prec, train_rec, train_f1 = train_scores
            test_scores = self.evaluate(y_test, y_preds)
            test_acc, test_ba, test_prec, test_rec, test_f1 = test_scores

            # Append to the results
            results.append([train_subject, *test_scores])

            print(f'Test subject {train_subject} --- Train Score: {train_acc}, {train_ba}, {train_prec}, {train_rec}, {train_f1} --- Test Score: {test_acc}, {test_ba}, {test_prec}, {test_rec}, {test_f1}')
        return results
    

    def subject_independent_evaluate(self, method: str) -> Dict[str, list]:
        logo = LeaveOneGroupOut()
        results = []

        for train_index, test_index in tqdm(logo.split(self.X, self.y, self.groups)):
            X_train, y_train, X_test, y_test = self.X[train_index], self.y[train_index], self.X[test_index], self.y[test_index]
            
            if self.__validate_label_cnt(y_train) == False or self.__validate_label_cnt(y_test) == False:
                continue

            clf = self.__get_classifier(method)
            _X_train, _X_test = self.__transform_data(method, X_train, X_test)
            # Train 
            clf.fit(_X_train, y_train)
            # Infer on test set
            y_preds = clf.predict(_X_test)
            y_preds_train = clf.predict(_X_train)
            # Evaluate
            train_scores = self.evaluate(y_train, y_preds_train)
            train_acc, train_ba, train_prec, train_rec, train_f1 = train_scores
            test_scores = self.evaluate(y_test, y_preds)
            test_acc, test_ba, test_prec, test_rec, test_f1 = test_scores

            # Append to the results
            test_subject = self.groups[test_index][0]
            results.append([test_subject, *test_scores])

            print(f'Test subject {test_subject} --- Train Score: {train_acc}, {train_ba}, {train_prec}, {train_rec}, {train_f1} --- Test Score: {test_acc}, {test_ba}, {test_prec}, {test_rec}, {test_f1}')
        return results


    def hybrid_evaluation(self, method: str) -> Dict[str, list]:
        pass


    def __validate_label_cnt(self, y):
        num_classes = len(np.unique(y))
        return num_classes > 1


    def exec_classifier(self):
        if self.subject_dependent is True:
            return self.subject_dependent_evaluate(self.strategy)
        if self.subject_independent is True:
            return self.subject_independent_evaluate(self.strategy)
        if self.hybrid_evaluation is True:
            return self.hybrid_evaluation(self.strategy)


    def evaluate(self, y_trues, y_preds):
        acc = accuracy_score(y_trues, y_preds)
        ba_acc = balanced_accuracy_score(y_trues, y_preds)
        prec = precision_score(y_trues, y_preds)
        rec = recall_score(y_trues, y_preds)
        f1 = f1_score(y_trues, y_preds)
        return [acc, ba_acc, prec, rec, f1]