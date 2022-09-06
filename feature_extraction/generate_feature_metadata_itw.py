import __init__ 
from datapath_manager import ITWDataPathManager,create_folder
import os
import argparse
import numpy as np
from date_time_utils import get_date_time_from_float
import pandas as pd
import itertools


class ITWMetadataGenerator:

    
    def __init__(self, dataset_name: str, window_size: int = 60, window_shift: float = 0.25):
        self.dataset_name = dataset_name
        self.dataset_path = ITWDataPathManager(self.dataset_name).get_dataset_path()
        self.window_size = window_size
        self.window_shift = window_shift


    def generate_metadata(self, user_id: str, date: str):
        """
        Generate metadata for a user including the following information:
            - date and time in float format
            - date and time of the recording of each session
        """

        user_feature_path = os.path.join(self.dataset_path, 'data', user_id, date, 'E4')
        assert(os.path.exists(user_feature_path))
        session_ids = sorted(os.listdir(user_feature_path))

        # Create output folder if it does not exist
        user_combined_feature_path = os.path.join(self.dataset_path, 'features', user_id, date)
        if not os.path.exists(user_combined_feature_path):
            create_folder(user_combined_feature_path)

        # Iterate through the statistical feature files in the folder and combine them
        SIGNALS = ['bvp', 'eda', 'temp']
        metadata = []
        for session_id in session_ids:
            print(f'Processing session: {session_id}')
            session_path = os.path.join(user_feature_path, session_id)
            _data = []
            for signal in SIGNALS:
                signal_path = os.path.join(session_path, f'{signal.upper()}.csv') 
                data = [float(line.rstrip()) for line in open(signal_path)]
                
                # Get the metadata from original recordings
                start_date_time = data[0]
                sampling_rate = data[1]

                # Compute recording length
                recording_length = len(data[2:]) - 1 # The last moment is not included
                first_iter = int(self.window_size * sampling_rate)
                step = int(self.window_shift * sampling_rate)
                recording_length = (recording_length - first_iter) // step + 1

                _data.append([start_date_time, recording_length])
            
            min_recording_length = np.argmin([d[1] for d in _data])
            start_date_time = _data[min_recording_length][0]
            dt_metadata = [start_date_time + i * self.window_shift for i in range(_data[min_recording_length][1])]
            dt_metadata = [(item, get_date_time_from_float(item)) for item in dt_metadata]
            metadata.append(dt_metadata)

        metadata = list(itertools.chain(*metadata))
        df_metadata = pd.DataFrame(metadata, columns=['date_time', 'date_time_str'])
        output_file_path = os.path.join(user_combined_feature_path, 'metadata.csv')  
        df_metadata.to_csv(output_file_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('--user_id', type=str, default=None)
    parser.add_argument('date', type=str, default=None)

    args = parser.parse_args()

    assert(args.user_id is not None)
    assert(args.date is not None)

    metadata_generator = ITWMetadataGenerator(args.dataset_name)

    metadata_generator.generate_metadata(args.user_id, args.date)