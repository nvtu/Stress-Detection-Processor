from re import S
from tkinter import W
import __init__
import os
from date_time_utils import convert_utc_to_local_time
from datapath_manager import ITWDataPathManager
import pandas as pd
import numpy as np
import argparse


class GenerateStressStateFromMoments:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.datapath_manager = ITWDataPathManager(dataset_name)


    def generate_stress_state(self, user_id: str, date: str, window_shift: float = 0.25):

        # List stress and relaxed moments
        user_date_path = self.datapath_manager.get_data_by_date_path(user_id, date)

        assert(os.path.exists(user_date_path))

        moments_path = os.path.join(user_date_path, 'Lifelog')
        relaxed_moments = [os.path.splitext(moment)[0] for moment in os.listdir(os.path.join(moments_path, 'Relaxed'))]
        stress_moments = [os.path.splitext(moment)[0] for moment in os.listdir(os.path.join(moments_path, 'Stress'))]

        # Merge relaxed and stress moments and order them in chronological order
        moments = [*stress_moments, *relaxed_moments]
        # labels = ['Stress', 'Relaxed']
        labels = [*[1 for _ in range(len(stress_moments))], *[0 for _ in range(len(relaxed_moments))]] # Stress = 1, Relaxed = 0
        
        chrono_moments = sorted(list(zip(moments, labels)), key=lambda x: x[0]) # Assume that the name of the moments is in chronological order
        num_moments = len(chrono_moments)
        
        WINDOW_LAG = 5 # Lagging in 5 seconds for uncontinuous stress & non-stress moments
        state_labels = []

        latest_moment = 0
        for i in range(num_moments):
            moment, label = chrono_moments[i]
            moment = convert_utc_to_local_time(moment).timestamp()
            prev_moment = moment - WINDOW_LAG
            next_moment = moment + WINDOW_LAG
            j = prev_moment
            extended_states = []
            while j < next_moment:
                if j >= latest_moment: # If the moment is not yet processed
                    extended_states.append((j, label))
                j += window_shift
            latest_moment = max(latest_moment, j) # Update the latest moment
            state_labels += extended_states
        

        
        # for i in range(1, num_moments):
        #     moment, label = chrono_moments[i]
        #     prev_moment, prev_label = chrono_moments[i-1]
        #     if label == prev_label:
        #         prev_moment = convert_utc_to_local_time(prev_moment).timestamp() # Convert to local time and add window shift as the previous moment is already labeled
        #         moment = convert_utc_to_local_time(moment).timestamp() # Convert to local time and add window shift as the current moment should have the same label
        #         j = prev_moment if i == 1 else prev_moment + window_shift
        #         while j <= moment:
        #             state_labels.append((j, label))
        #             j += window_shift
        #     else: 
        #         # If the current moment is not the same as the previous moment, then there is a gap between the two moments by at least 10 seconds
        #         # We need to label the gap as the previous label
        #         prev_moment = convert_utc_to_local_time(prev_moment).timestamp()
        #         j = prev_moment if i == 1 else prev_moment + window_shift
        #         while j < prev_moment + WINDOW_LAG:
        #             state_labels.append((j, prev_label))
        #             j += window_shift
        #         # And the other 10 seconds as the current label
        #         moment = convert_utc_to_local_time(moment).timestamp()
        #         j = moment - WINDOW_LAG
        #         while j <= moment:
        #             state_labels.append((j, label))
        #             j += window_shift

        return state_labels

    
    def generate_moments_by_metadata(self, user_id: str, date: str, window_shift: float = 0.25):
        state_labels = self.generate_stress_state(user_id, date, window_shift)
        feature_stats_path, metadata_path = self.datapath_manager.get_feature_file_path_by_date(user_id, date)

        assert(os.path.exists(metadata_path))
        assert(os.path.exists(feature_stats_path))

        metadata = pd.read_csv(metadata_path)
        features = np.load(feature_stats_path)

        state_datetime = pd.DataFrame([item[0] for item in state_labels], columns=['date_time'])
        # metadata_datetime = metadata['date_time'].to_list()
        # print(metadata_datetime)
        # Find the intersection of the state labels and the metadata
        feature_indices = metadata[metadata['date_time'].isin(state_datetime['date_time'])].index.tolist()

        feature_moments = metadata['date_time'].iloc[feature_indices].to_list()

        # a = sorted(state_datetime[state_datetime['date_time'].isin(feature_moments)]['date_time'].tolist())
        # b = [i for i in range(len(a)) if a[i] == a[i-1]]
        # print(a[8], a[7])
        # print(b)
        # print(len(a))
        # print(len(list(set(a))))
        # print(len(feature_moments))
        label_indices = state_datetime[state_datetime['date_time'].isin(feature_moments)].index.tolist()

        stress_state_features = features[feature_indices]

        stress_state_labels = np.array(state_labels)[label_indices][:, 1] # Get the labels only
        
        assert(stress_state_features.shape[0] == stress_state_labels.shape[0])

        # Save the stress state labels
        feature_path = self.datapath_manager.get_feature_path_by_date(user_id, date)
        stress_state_path = os.path.join(feature_path, 'y.npy')
        stress_state_feature_path = os.path.join(feature_path, 'X.npy')
        np.save(stress_state_path, stress_state_labels)
        np.save(stress_state_feature_path, stress_state_features)

    
    def generate_stress_state_by_moments(self, user_id: str, date: str):
        """
        Generate stress state by image moment information
        """

        # List stress and relaxed moments
        user_date_path = self.datapath_manager.get_data_by_date_path(user_id, date)

        moments_path = os.path.join(user_date_path, 'Lifelog')
        relaxed_moments = [os.path.splitext(moment)[0] for moment in os.listdir(os.path.join(moments_path, 'Relaxed'))]
        stress_moments = [os.path.splitext(moment)[0] for moment in os.listdir(os.path.join(moments_path, 'Stress'))]

        features_stats_path, metadata_path = self.datapath_manager.get_feature_file_path_by_date(user_id, date)

        assert(os.path.exists(user_date_path))
        assert(os.path.exists(metadata_path))
        assert(os.path.exists(features_stats_path))

        features = np.load(features_stats_path)

        moments = [*stress_moments, *relaxed_moments]
        # timestamp_moments = [convert_utc_to_local_time(moment).timestamp() for moment in moments]
        labels = [*[1 for _ in range(len(stress_moments))], *[0 for _ in range(len(relaxed_moments))]] # Stress = 1, Relaxed = 0

        metadata = pd.read_csv(metadata_path)
        metadata_datetime = metadata['date_time'].to_list()

        chrono_moments = sorted(list(zip(moments, labels)), key=lambda x: x[0]) # Assume that the name of the moments is in chronological order
        timestamp_moments = [convert_utc_to_local_time(x[0]).timestamp() for x in chrono_moments]
        print(chrono_moments)
        # Find the intersection of the state labels and the metadata
        intersection_indices = metadata[metadata['date_time'].isin(timestamp_moments)].index.tolist()
        # print(metadata.iloc[intersection_indices]['date_time'].tolist())
        filtered_features = features[intersection_indices]

        chrono_intersection_indices = [i for i in range(len(timestamp_moments)) if timestamp_moments[i] in metadata_datetime]
        filtered_moments = np.array(chrono_moments)[chrono_intersection_indices]
        # print(filtered_moments)

        datetime_info = np.array(filtered_moments[:, 0])
        labels = np.array(filtered_moments[:, 1]).astype(int)

        feature_path = self.datapath_manager.get_feature_path_by_date(user_id, date)
        datetime_path = os.path.join(feature_path, 'datetime_info.npy')
        ground_truth_path = os.path.join(feature_path, 'y_moment.npy')
        feature_path = os.path.join(feature_path, 'X_moment.npy')

        np.save(datetime_path, datetime_info)
        np.save(ground_truth_path, labels)
        np.save(feature_path, filtered_features)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='DCU_EXP2_ITW', help='Name of the dataset')
    parser.add_argument('--user_id', type=str, default='nvtu', help='User ID')
    parser.add_argument('--date', type=str, default='2022-09-02', help='Date')
    parser.add_argument('--window_shift', type=float, default=0.25, help='Window shift in seconds')
    args = parser.parse_args()

    generate_stress_state = GenerateStressStateFromMoments(args.dataset_name)
    generate_stress_state.generate_stress_state_by_moments(args.user_id, args.date) 
    generate_stress_state.generate_moments_by_metadata(args.user_id, args.date, args.window_shift)