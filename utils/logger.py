from mlflow import log_metrics, log_param, log_params
from typing import List, Optional, Dict, Any


class Logger:

    """
    A logger helper for all kinds of logging the data process
    """

    def __init__(self, save_log_path: str):
        self.save_log_path = save_log_path

    
    def append(self, log_str: str) -> None:
        with open(self.save_log_path, 'a') as f:
            print(log_str, file=f)
    

    def write(self, log_str: str) -> None:
        with open(self.save_log_path, 'w') as f:
            print(log_str, file=f)
