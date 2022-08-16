import os
import sys

data_lib = os.path.abspath('../data_processing')

if data_lib not in sys.path:
    sys.path.append(data_lib)