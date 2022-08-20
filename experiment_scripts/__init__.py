import os
import sys


parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_folder not in sys.path:
    sys.path.append(parent_folder)

paths = [os.path.join(parent_folder, 'data_processing'),
    os.path.join(parent_folder, 'models'),
    os.path.join(parent_folder, 'feature_extraction'),
    os.path.join(parent_folder, 'signal_processing'),
    os.path.join(parent_folder, 'utils'),
]

for path in paths:
    if path not in sys.path:
        sys.path.append(path)
    
