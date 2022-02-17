import numpy as np
from scipy.stats import linregress


class TEMP_Signal_Processor:

    def temp_feature_extraction(self, temp):
        len_signal = len(temp)
        t = np.linspace(0, len_signal, len_signal)
        mean_temp, std_temp = temp.mean(), temp.std()
        min_temp, max_temp = temp.min(), temp.max()
        range_temp = max_temp - min_temp
        res = linregress(t, temp)
        temp_slope = res.slope
        features = [mean_temp, std_temp, min_temp, max_temp, range_temp, temp_slope]
        return features