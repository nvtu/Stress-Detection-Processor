from collections import defaultdict
from datapath_manager import *
from tqdm import tqdm


def preprocess_dataset(dataset_name: str):
    """
    Preprocess the whole dataset.
    The dataset structure should be like:
        dataset_name
            |- user_id
                |- acc
                |- bvp
                |- eda
                |- temp
    """

    eda = defaultdict(list)
    bvp = defaultdict(list)
    temp = defaultdict(list)
    
    dp_manager = get_datapath_manager(dataset_name)
    for user_id, user_data_path in tqdm(dp_manager.user_data_paths.items()):
        pass


def preprocess_user_data(dataset_name: str, user_id: str):
    pass

