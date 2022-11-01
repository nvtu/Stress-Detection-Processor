import __init__ 
from datapath_manager import ITWDataPathManager,create_folder
import os
import argparse
import numpy as np



class ITWFeatureCombiner:

    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_path = ITWDataPathManager(self.dataset_name).get_dataset_path()


    def combine_session_features(self, user_id: str, date: str, session_id: str):
        """
        Combine statistical features for session
        The function iterates through the statistical feature files in a folder and combines them
        The iteration order follows the lexicographical order of the file names
        NOTE: Check the configuration files of the BranchNeuralNetwork class to see which features are used and ensure the correct dimensions are used for each branch
        """

        user_feature_path = os.path.join(self.dataset_path, 'data', user_id, date, 'E4')
        assert(os.path.exists(user_feature_path))
        session_feature_path = os.path.join(user_feature_path, session_id, 'features')


        # Iterate through the statistical feature files in the folder and combine them
        features = []
        for feature_file in sorted(os.listdir(session_feature_path)):
            feature_path = os.path.join(session_feature_path, feature_file)
            feat = np.load(feature_path)
            features.append(feat)
        min_num_features = min([feat.shape[0] for feat in features]) 
        features = [feat[:min_num_features, :] for feat in features]
        features = np.concatenate(features, axis=1)
        return features

    
    def combine_features(self, user_id: str, date: str):
        """
        Combine statistical features for training
        The function iterates through the statistical feature files in a folder and combines them
        The iteration order follows the lexicographical order of the file names
        NOTE: Check the configuration files of the BranchNeuralNetwork class to see which features are used and ensure the correct dimensions are used for each branch
        """

        user_feature_path = os.path.join(self.dataset_path, 'data', user_id, date, 'E4')
        assert(os.path.exists(user_feature_path))
        session_ids = sorted(os.listdir(user_feature_path))

        # Create output folder if it does not exist
        user_combined_feature_path = os.path.join(self.dataset_path, 'features', user_id, date)
        if not os.path.exists(user_combined_feature_path):
            create_folder(user_combined_feature_path)

        # Iterate through the statistical feature files in the folder and combine them
        combined_features = []
        for session_id in session_ids:
            print(f'Processing session: {session_id}')
            features = self.combine_session_features(user_id, date, session_id)
            combined_features.append(features)

        combined_features = np.vstack(combined_features)

        # Filter the NaN features
        user_date_path = os.path.dirname(user_feature_path)
        removed_indices = list(map(int, [line.strip() for line in open(os.path.join(user_date_path, 'itw_logs.txt'), 'r').readlines()]))
        mask = np.ones(combined_features.shape[0], dtype=bool)
        mask[removed_indices] = False
        combined_features = combined_features[mask, :]
        

        output_file_path = os.path.join(user_combined_feature_path, 'bvp_eda_temp.npy')  
        np.save(output_file_path, combined_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('--user_id', type=str, default=None)
    parser.add_argument('date', type=str, default=None)

    args = parser.parse_args()

    assert(args.user_id is not None)
    assert(args.date is not None)

    feature_combiner = ITWFeatureCombiner(args.dataset_name)

    feature_combiner.combine_features(args.user_id, args.date)