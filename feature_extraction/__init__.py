import os
import sys

data_lib = os.path.abspath('../data_processing')
signal_processing_lib = os.path.abspath('../signal_processing')


if data_lib not in sys.path:
    sys.path.append(data_lib)
if signal_processing_lib not in sys.path:
    sys.path.append(signal_processing_lib)