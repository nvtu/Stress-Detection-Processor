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
                    random_state: int = 0, cross_validation: bool = False, logo_validation: bool = False, basic_logo_validation: bool = False, scoring = 'accuracy'):
        self.X = X
        self.y = y
        self.strategy = strategy # Stress detection strategy - possible options: mlp, knn, svm, logistic_regression, random_forest
        self.groups = groups
        self.random_state = random_state
        self.cross_validation = cross_validation # Cross-validation approach
        self.logo_validation = logo_validation # Leave one group out approach
        self.basic_logo_validation = basic_logo_validation
        self.scoring = scoring
    

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
    

    def split_train_test_cv(self, test_size = 0.2):
        X_train = np.array([])
        X_test = np.array([])
        y_train = np.array([])
        y_test = np.array([])
        num_items = len(self.y)
        first_pointer = 0
        train_size = 1 - test_size
        for i in range(1, num_items):
            if self.y[i] != self.y[i-1] or i == num_items - 1:
                if i == num_items - 1: i += 1 
                _y = self.y[first_pointer:i]
                _X = self.X[first_pointer:i]
                train_index = int(train_size * len(_y))
                X_train = np.append(X_train, _X[:train_index])
                y_train = np.append(y_train, _y[:train_index])
                X_test = np.append(X_test, _X[train_index:])
                y_test = np.append(y_test, _y[train_index:])
                first_pointer = i
        X_train = X_train.reshape(len(y_train), -1)
        y_train = np.array(y_train)
        X_test = X_test.reshape(len(y_test), -1)
        y_test = np.array(y_test)
        return X_train, X_test, y_train, y_test


    def cross_validator(self, method: str):
        # X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.286, stratify = self.y, random_state = self.random_state) # Split train test data
        # X_train, X_test, y_train, y_test = self.split_train_test_cv(test_size = 0.286)
        # Validate if the data has two classes
        # num_classes = len(np.unique(self.y)
        logo = LeaveOneGroupOut()
        balanced_accs = []
        test_groups = []
        for _, test_index in tqdm(logo.split(self.X, self.y, self.groups)):
            user_dataset, user_ground_truth = self.X[test_index], self.y[test_index]
        
            # Validate if the test set and train set have enough classes
            num_classes_test = len(np.unique(user_ground_truth))
            if num_classes_test < 2:
                continue

            skf = StratifiedKFold(n_splits=3)
            test_scores = []
            for _tr_index, _t_index in skf.split(user_dataset, user_ground_truth):
                pipeline = []
                if method in ['mlp', 'svm', 'knn', 'Voting3CLF']:
                    pipeline.append(('sc', StandardScaler()))
                estimator = self.__get_classifier(method)
                pipeline.append(('estimator', estimator))
                pipeline = Pipeline(pipeline)
            # scores = cross_validate(pipeline, X=user_dataset, y=user_ground_truth, scoring=self.scoring, cv=skf)
                pipeline.fit(user_dataset[_tr_index], user_ground_truth[_tr_index])
                y_preds = pipeline.predict(user_dataset[_t_index])
                bacc_score = self.evaluate(user_ground_truth[_t_index], y_preds)
                test_scores.append(bacc_score)
            print(f'{self.groups[test_index][0]} ---', test_scores, np.mean(test_scores))            
            balanced_accs.append(np.mean(test_scores))
            # balanced_accs.append(np.mean(scores['test_score']))
            test_groups.append(self.groups[test_index][0])
        results = { 'groups': test_groups, 'balanced_accuracy_score': balanced_accs }
        return results
       

    def basic_logo_validator(self, method: str) -> Dict[str, list]:
        logo = LeaveOneGroupOut()
        test_groups = []
        scores = []

        for train_index, test_index in tqdm(logo.split(self.X, self.y, self.groups)):
            X_train, y_train, X_test, y_test = self.X[train_index], self.y[train_index], self.X[test_index], self.y[test_index]
            clf = self.__get_classifier(method)
            _X_train, _X_test = self.__transform_data(method, X_train, X_test)
            # Train 
            clf.fit(_X_train, y_train)
            # Infer on test set
            y_preds = clf.predict(_X_test)
            y_preds_train = clf.predict(_X_train)
            train_scores = self.evaluate(y_train, y_preds_train)
            train_acc, train_ba, train_prec, train_rec, train_f1 = train_scores
            test_scores = self.evaluate(y_test, y_preds)
            test_acc, test_ba, test_prec, test_rec, test_f1 = test_scores

            # Append to the results
            scores.append(test_scores)
            test_subject = self.groups[test_index][0]
            test_groups.append(test_subject)

            print(f'Test subject {test_subject} --- Train Score: {train_acc}, {train_ba}, {train_prec}, {train_rec}, {train_f1} --- Test Score: {test_acc}, {test_ba}, {test_prec}, {test_rec}, {test_f1}')
        results = list(zip(test_groups, scores))
        print(results)
        return results


    def __validate_label_cnt(y):
        num_classes = len(np.unique(y))
        return num_classes > 1


    def leave_one_group_out_validator(self, method: str) -> Dict[str, list]:
        test_groups = []
        balanced_accs = []
        accs = []
        cv_balanced_acc_scores = []


        for train_index, test_index in tqdm(LeaveOneGroupOut().split(self.X, self.y, self.groups)):
            X_train, y_train, X_test, y_test = self.X[train_index], self.y[train_index], self.X[test_index], self.y[test_index] # Get train and test data
            train_groups = self.groups[train_index]

            # Validate if the test set and train set have two classes
            if self.__validate_label_cnt(y_train) == False or self.__validate_label_cnt(y_test) == False: # If one of them does not have enough classes, then ignore it
                continue

            preds = []
            for _train_index, validate_index in LeaveOneGroupOut().split(X_train, y_train, train_groups):
                _X, _y, X_val, y_val = X_train[_train_index], y_train[_train_index], X_train[validate_index], y_train[validate_index]

                # Validate if the test set and train set have two classes
                if self.__validate_label_cnt(_y) == False or self.__validate_label_cnt(y_val) == False: # If one of them does not have enough classes, then ignore it
                    continue

                clf = self.__get_classifier(method)
                __X, _X_test = self.__transform_data(method, _X, X_test) # Feature scaling if possible
                _, X_val = self.__transform_data(method, _X, X_val)
                clf.fit(__X, _y)
                y_val_pred = clf.predict(X_val)

                # Run prediction on test set
                y_preds = clf.predict(_X_test)
                # print(f'--- Validate {train_groups[validate_index][0]} BA Score: {self.evaluate(y_val, y_val_pred, self.scoring)} --- Test BA Score: {self.evaluate(y_test, y_preds, self.scoring)}')
                preds.append(y_preds)

            preds = np.mean(np.array(preds), axis=0)
            y_preds = np.array([0 if value < 0.5 else 1 for value in preds])                    

            # Evaluate balanced accuracy on the predicted results of test set
            balanced_accuracy = self.evaluate(y_test, y_preds, self.scoring)
            accuracy = self.evaluate(y_test, y_preds, 'accuracy')
            balanced_accs.append(balanced_accuracy) 
            accs.append(accuracy)

            # Save the corresponding user_id
            test_groups.append(self.groups[test_index][0])
            print(f'--- Test Group {self.groups[test_index][0]} BA Score: {balanced_accuracy} --- Test Acc Score: {accuracy} --- Train BA Score {self.evaluate(y_train, clf.predict(X_train), self.scoring)} ---')
        
        results = { 'groups': test_groups, 'balanced_accuracy_score': balanced_accs, 'accuracy_score': accs }
        return results


    def train_and_infer(self, X_test, y_test):
        clf = self.__get_classifier(self.strategy)
        X_train, X_test = self.__transform_data(self.strategy, self.X, X_test)
        clf.fit(X_train, self.y)
        y_preds = clf.predict(X_test)
        balanced_accuracy = self.evaluate(y_test, y_preds)
        return balanced_accuracy


    def exec_classifier(self):
        if self.cross_validation is True:
            return self.cross_validator(self.strategy)
        if self.logo_validation is True:
            return self.leave_one_group_out_validator(self.strategy)
        if self.basic_logo_validation is True:
            return self.basic_logo_validator(self.strategy)


    def evaluate_score(self, y_trues, y_preds, scoring):
        acc = None
        if scoring == 'accuracy':
            acc = accuracy_score(y_trues, y_preds)
        elif scoring == 'balanced_accuracy':
            acc = balanced_accuracy_score(y_trues, y_preds)
        return acc              

    
    def evaluate(self, y_trues, y_preds):
        acc = accuracy_score(y_trues, y_preds)
        ba_acc = balanced_accuracy_score(y_trues, y_preds)
        prec = precision_score(y_trues, y_preds)
        rec = recall_score(y_trues, y_preds)
        f1 = f1_score(y_trues, y_preds)
        return [acc, ba_acc, prec, rec, f1]