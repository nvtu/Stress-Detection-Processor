from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score, auc
from typing import List, Optional


class Evaluator:
    """
    Helper for evaluating the performance of a model on a dataset using a set of metrics
    """

    def __init__(self, target_metrics: List[str] = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']):
        self.target_metrics = target_metrics


    def evaluate_on_metrics(self, y_true, y_pred, metric_name: str) -> Optional[float]:
        if metric_name == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif metric_name == 'balanced_accuracy':
            return balanced_accuracy_score(y_true, y_pred)
        elif metric_name == 'precision':
            return precision_score(y_true, y_pred)
        elif metric_name == 'recall':
            return recall_score(y_true, y_pred)
        elif metric_name == 'f1':
            return f1_score(y_true, y_pred)
        elif metric_name == 'auc':
            # fpr, tpr, _ = roc_curve(y_true, y_pred)
            # return auc(fpr, tpr)
            return roc_auc_score(y_true, y_pred)
        return None


    def evaluate(self, y_true, y_pred) -> List[float]:
        scores = {}
        for metrics in self.target_metrics:
            score = self.evaluate_on_metrics(y_true, y_pred, metrics)
            scores[metrics] = score
        return scores