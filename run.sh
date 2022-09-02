dataset_name=$1
model_type=$2

#python run_classifier.py $dataset_name --model_name knn --detector_type $model_type --window_size 60 --window_shift 0.25
#python run_classifier.py $dataset_name --model_name lda --detector_type $model_type --window_size 60 --window_shift 0.25
#python run_classifier.py $dataset_name --model_name logistic_regression --detector_type $model_type --window_size 60 --window_shift 0.25
#python run_classifier.py $dataset_name --model_name random_forest --detector_type $model_type --window_size 60 --window_shift 0.25
#python run_classifier.py $dataset_name --model_name gradient_boosting --detector_type $model_type --window_size 60 --window_shift 0.25
#python run_classifier.py $dataset_name --model_name svm --detector_type $model_type --window_size 60 --window_shift 0.25
python run_classifier.py $dataset_name --model_name extra_trees --detector_type $model_type --window_size 60 --window_shift 0.25
