import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.stats.mstats import winsorize
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_frequency_domain_features, get_time_domain_features, get_geometrical_features, get_poincare_plot_features
from scipy.stats import kurtosis, skew



class BVP_Signal_Processor:


    def winsorization(self, bvp, limit=0.02):
        """
        The winsorization method is used to remove outliers from a signal.
        The originality of this method for BVP processing is from:
        https://www.researchgate.net/publication/308611962_CONTINUOUS_STRESS_MONITORING_USING_A_WRIST_DEVICE_AND_A_SMARTPHONE
        """
        output = winsorize(bvp, limits = [limit, limit])
        return output


    def butter_baseline_drift_removal(self, bvp, sampling_rate):
        """
        The idea of bvp baseline drift removal using Butterworth filter follows the methods described in 
        http://www.jscholaronline.org/articles/JBER/Signal-Processing.pdf
        """
        BUTTERWORTH_ORDER = 4
        CUTOFF_FREQUENCY = 0.5
        b, a = butter(BUTTERWORTH_ORDER, CUTOFF_FREQUENCY, btype = 'high', fs = sampling_rate)
        output = filtfilt(b, a, bvp)
        return output

    
    def min_max_norm(self, bvp):
        output = MinMaxScaler().fit_transform(bvp.reshape(-1, 1)).ravel()
        return output
    

    def standard_norm(self, bvp):
        output = StandardScaler().fit_transform(bvp.reshape(-1, 1)).ravel()
        return output
    

    # ---------------------- HRV UTILS FROM NEUROKIT2 ----------------------------------

    def _hrv_get_rri(self, peaks = None, sampling_rate = 1000, interpolate = False, **kwargs):
        rri = np.diff(peaks) / sampling_rate * 1000
        if interpolate is False:
            return rri
        else:
            # Minimum sampling rate for interpolation
            if sampling_rate < 10:
                sampling_rate = 10
            # Compute length of interpolated heart period signal at requested sampling rate.
            desired_length = int(np.rint(peaks[-1]))
            rri = nk.signal_interpolate(
                peaks[1:],  # Skip first peak since it has no corresponding element in heart_period
                rri,
                x_new=np.arange(desired_length),
                **kwargs
            )
            return rri, sampling_rate


    def _hrv_sanitize_input(self, peaks=None):

        if isinstance(peaks, tuple):
            peaks = self._hrv_sanitize_tuple(peaks)
        elif isinstance(peaks, (dict, pd.DataFrame)):
            peaks = self._hrv_sanitize_dict_or_df(peaks)
        else:
            peaks = self._hrv_sanitize_peaks(peaks)

        return peaks


    # =============================================================================
    # Internals
    # =============================================================================
    def _hrv_sanitize_tuple(self, peaks):

        # Get sampling rate
        info = [i for i in peaks if isinstance(i, dict)]
        sampling_rate = info[0]['sampling_rate']

        # Get peaks
        if isinstance(peaks[0], (dict, pd.DataFrame)):
            try:
                peaks = self._hrv_sanitize_dict_or_df(peaks[0])
            except NameError:
                if isinstance(peaks[1], (dict, pd.DataFrame)):
                    try:
                        peaks = self._hrv_sanitize_dict_or_df(peaks[1])
                    except NameError:
                        peaks = self._hrv_sanitize_peaks(peaks[1])
                else:
                    peaks = self._hrv_sanitize_peaks(peaks[0])

        return peaks, sampling_rate


    def _hrv_sanitize_dict_or_df(self, peaks):

        # Get columns
        if isinstance(peaks, dict):
            cols = np.array(list(peaks.keys()))
            if 'sampling_rate' in cols:
                sampling_rate = peaks['sampling_rate']
            else:
                sampling_rate = None
        elif isinstance(peaks, pd.DataFrame):
            cols = peaks.columns.values
            sampling_rate = None

        cols = cols[["Peak" in s for s in cols]]

        if len(cols) > 1:
            cols = cols[[("ECG" in s) or ("PPG" in s) for s in cols]]

        if len(cols) == 0:
            raise NameError(
                "NeuroKit error: hrv(): Wrong input, ",
                "we couldn't extract R-peak indices. ",
                "You need to provide a list of R-peak indices.",
            )

        peaks = self._hrv_sanitize_peaks(peaks[cols[0]])

        if sampling_rate is not None:
            return peaks, sampling_rate
        else:
            return peaks


    def _hrv_sanitize_peaks(self, peaks):

        if isinstance(peaks, pd.Series):
            peaks = peaks.values

        if len(np.unique(peaks)) == 2:
            if np.all(np.unique(peaks) == np.array([0, 1])):
                peaks = np.where(peaks == 1)[0]

        if isinstance(peaks, list):
            peaks = np.array(peaks)

        return peaks

    # ---------------------------------------------------------------------------------------------


    def clean_bvp(self, bvp, sampling_rate):
        cleaned_bvp = self.winsorization(bvp)
        cleaned_bvp = self.butter_baseline_drift_removal(cleaned_bvp, sampling_rate=sampling_rate)

        return cleaned_bvp

    
    def clean_rr_intervals(self, rr_intervals):
        rr_intervals_list = rr_intervals

        # This remove outliers from signal
        rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals_list,  verbose = False,
                                                        low_rri=300, high_rri=2000)
        # This replace outliers nan values with linear interpolation
        interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers, 
                                                        interpolation_method="linear")

        # This remove ectopic beats from signal
        nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik", verbose=False)
        
        # This replace ectopic beats nan values with linear interpolation
        interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
        return np.array(interpolated_rr_intervals), interpolated_nn_intervals


    def extract_rr_intervals(self, bvp, sampling_rate):
        ppg_signals, info = nk.ppg_process(bvp, sampling_rate = sampling_rate)
        hr = ppg_signals['PPG_Rate']
        peaks = info['PPG_Peaks']

        # Sanitize input
        peaks = self._hrv_sanitize_input(peaks)
        if isinstance(peaks, tuple):  # Detect actual sampling rate
            peaks, sampling_rate = peaks[0], peaks[1]
        rri = self._hrv_get_rri(peaks, sampling_rate = sampling_rate, interpolate = False)
        return rri

    
    def bvp_feature_extraction(self, bvp, sampling_rate):
        rri = self.extract_rr_intervals(bvp, sampling_rate)
        interpolated_rr_intervals, interpolated_nn_intervals = self.clean_rr_intervals(rri)
        time_domain_features = get_time_domain_features(interpolated_nn_intervals)
        frequency_domain_features = get_frequency_domain_features(interpolated_nn_intervals, sampling_frequency = sampling_rate, method = 'welch')
        geometrical_features = get_geometrical_features(interpolated_nn_intervals)
        pointcare_features = get_poincare_plot_features(interpolated_nn_intervals)

        # Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, and Kristof Van Laerhoven. 2018. Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.
        # In Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI '18). Association for Computing Machinery, New York, NY, USA, 400–408. DOI:https://doi.org/10.1145/3242969.3242985
        # Not including: f_x_HRV of ULF and HLF, rel_f_x, sum f_x_HRV

        mean_HR, std_HR = time_domain_features['mean_hr'], time_domain_features['std_hr']
        mean_HRV, std_HRV = time_domain_features['mean_nni'], time_domain_features['sdnn']
        HRV_VLF, HRV_LF, HRV_HF, HRV_LFHF, HRV_LFnorm, HRV_HFnorm = frequency_domain_features['vlf'], frequency_domain_features['lf'], frequency_domain_features['hf'], frequency_domain_features['lf_hf_ratio'], frequency_domain_features['lfnu'], frequency_domain_features['hfnu']
        nn50, HRV_pNN50 = time_domain_features['nni_50'], time_domain_features['pnni_50']
        nn20, HRV_pNN20 = time_domain_features['nni_20'], time_domain_features['pnni_20']
        HRV_RMSSD = time_domain_features['rmssd']
        total_power = frequency_domain_features['total_power']
        rms = np.sqrt(np.mean(interpolated_rr_intervals ** 2))
        HRV_HTI = geometrical_features['triangular_index']

        # Nkurikiyeyezu, K., Yokokubo, A., & Lopez, G. (2019). The Influence of Person-Specific Biometrics in Improving Generic Stress Predictive Models. 
        # ArXiv, abs/1910.01770.
        kurtosis_HRV, skewness_HRV = kurtosis(interpolated_rr_intervals), skew(interpolated_rr_intervals)
        HRV_SD1, HRV_SD2 = pointcare_features['sd1'], pointcare_features['sd2']
        HRV_SDSD = time_domain_features['sdsd']
        HRV_SDSD_RMSSD = HRV_SDSD / HRV_RMSSD
        adj_sum_rri = np.diff(interpolated_rr_intervals) + 2 * interpolated_rr_intervals[:-1]
        relative_RRI = 2 * np.diff(interpolated_rr_intervals) / adj_sum_rri
        mean_relativeRRI, median_relativeRRI, std_relativeRRI, RMSSD_relativeRRI, kurtosis_relativeRRI, skew_relativeRRI = np.mean(relative_RRI), np.median(relative_RRI), np.std(relative_RRI), np.sqrt(np.mean(np.diff(relative_RRI) ** 2)), kurtosis(relative_RRI), skew(relative_RRI)
        # Combining the extracted features
        features = [mean_HR, std_HR, mean_HRV, std_HRV, kurtosis_HRV, skewness_HRV, rms, nn50, HRV_pNN50, nn20, HRV_pNN20, HRV_HTI, total_power, HRV_RMSSD, HRV_VLF, HRV_LF, HRV_HF, HRV_LFHF, HRV_LFnorm, HRV_HFnorm, HRV_SD1, HRV_SD2, HRV_SDSD, HRV_SDSD_RMSSD, mean_relativeRRI, median_relativeRRI, std_relativeRRI, RMSSD_relativeRRI, kurtosis_relativeRRI, skew_relativeRRI]
        features = np.array(list(map(float, features)))
        return features