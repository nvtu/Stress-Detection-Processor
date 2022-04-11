from datapath_manager import DataPathManager, create_folder
import os
import pandas as pd


class ResultUtils:

    """
    Utility to help dump the prediction results to the csv file
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dp_manager = DataPathManager(dataset_name)
    

    def dump_result_to_csv(self, results, columns):
        output_folder = os.path.join(self.dp_manager.dataset_path, 'results')
        create_folder(output_folder)
        output_file_path = os.path.join(output_folder, f'{self.dataset_name}.csv')
        df = pd.DataFrame(results, columns = columns)
        df.to_csv(output_file_path, index = False)
