# STRESS DETECTION PIPELINE WITH PHYSIOLOGICAL SIGNALS

The Stress Detection Pipeline comprises of two following steps:

1. [Bio-signal Processing and Statistical Feature Processing](#bio)
2. [Train stress detection model](#train)
3. [Use the trained model to predict other data](#infer)

# 1. Bio-signal Processing & Statistical Feature Extraction
This process can be run in parallel to boost the speed of the statistical feature extraction by running the command-line separately for each physiological signal. **The bio-signal processing is included in the feature extraction progress**. <a name="bio"></a>
```
$ python extract_features.py --dataset_name [DATASET_NAME] --signal [SIGNAL] --window_size [WINDOW_SIZE] --window_shift [WINDOW_SHIFT]
```
After all the corresponding features are extracted, running this command-line to combine the **features** for training:
```
$ python combine_features.py --dataset_name [DATASET_NAME] --window_size [WINDOW_SIZE] --window_shift [WINDOW_SHIFT]
```
Finally, generate the metadata which includes **ground_truth** and **groups** for subject-independent/subject-dependent training:
```
$ python generate_feature_metadata.py --dataset_name [DATASET_NAME] --window_size [WINDOW_SIZE] --window_shift [WINDOW_SHIFT]
```
If you only want to process the bio-signal and extract features for a user's data:
```
$ python extract_features.py --dataset_name [DATASET_NAME] --signal [SIGNAL] --window_size [WINDOW_SIZE] --window_shift [WINDOW_SHIFT] --user_id [USER_ID]
$ python generate_feature_metadata.py --dataset_name [DATASET_NAME] --window_size [WINDOW_SIZE] --window_shift [WINDOW_SHIFT] --user_id [USER_ID]
```
In this step, the aforementioned parameters in the command-lines are as follows:
- DATASET_NAME: WESAD, AffectiveROAD, CognitiveDS, DCU_NVT_EXP2
- SIGNAL: eda, bvp, temp
- WINDOW_SIZE: 60 (by default as frequently used by many related paper)
- WINDOW_SHIFT: 0.25 (by default as frequently used by many related paper)


# 2. Train Stress Detection Model <a name="train"></a> 
Train the stress detection model using the command-line with suitable parameters:
```
$ python run_classifier.py --dataset_name [DATASET_NAME] --model_name [MODEL_NAME] --detector_type [DETECTOR_TYPE] --window_size [WINDOW_SIZE] --window_shift [WINDOW_SHIFT]
```
In this step, the aforementioned parameters in the command-lines are as follows:
- **DATASET_NAME**: WESAD, AffectiveROAD, CognitiveDS, DCU_NVT_EXP2
- **MODEL_NAME**: random_forest, knn, svm, branch_neural_network
- **DETECTOR_TYPE**: General (Subject-Independent Model), Personal (Subject-Dependent Model)
- **WINDOW_SIZE**: 60 (by default as frequently used by many related paper)
- **WINDOW_SHIFT**: 0.25 (by default as frequently used by many related paper)

If you only want to train the subject-dependent model for a target user:
```
$ python run_classifier.py --dataset_name [DATASET_NAME] --model_name [MODEL_NAME] --detector_type Personal --window_size [WINDOW_SIZE] --window_shift [WINDOW_SHIFT] --user_id [USER_ID]
```