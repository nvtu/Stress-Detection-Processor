from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import numpy as np
import pywt
import pandas as pd
import neurokit2 as nk
import scipy
from scipy.stats import kurtosis, skew, linregress, pearsonr
import math


class SWT_Threshold_Denoiser:


    def __compute_gmm_cdf(self, x, weights, means, covars):
        n_components = len(weights)
        mcdf = 0
        for i in range(n_components):
            mcdf += weights[i] * norm.cdf(x, loc=means[i], scale=covars[i])
        return mcdf[0][0]

    
    def __find_inverse_gmm_cdf(self, cdf_value, gmm, max_value, min_value):
        EPS = 1e-9
        if min_value * max_value < 0:
            x_pos = self.__find_inverse_gmm_cdf(cdf_value, gmm, max_value, 0)
            x_neg = self.__find_inverse_gmm_cdf(cdf_value, gmm, 0, min_value)
            cdf_pos = self.__compute_gmm_cdf(x_pos, gmm.weights_, gmm.means_, gmm.covariances_)
            cdf_neg = self.__compute_gmm_cdf(x_neg, gmm.weights_, gmm.means_, gmm.covariances_)
            diff_cdf_pos = abs(cdf_pos - cdf_value)
            diff_cdf_neg = abs(cdf_neg-cdf_value)
            x = x_pos if diff_cdf_pos <= diff_cdf_neg else x_neg
        else:
            while max_value - min_value > EPS:
                x = (max_value + min_value) / 2
                cdf = self.__compute_gmm_cdf(x, gmm.weights_, gmm.means_, gmm.covariances_)
                if cdf <= cdf_value:
                    min_value = x
                elif cdf > cdf_value:
                    max_value = x
                if abs((max_value + min_value) / 2 - x) <= EPS: break
            x = (max_value + min_value) / 2
        return x
    

    def __swt_thresholding_denoise(self, cDn, artifact_proportion):
        gmm = GaussianMixture(n_components = 2, random_state = 0).fit(cDn.reshape(-1, 1))
        mmax, mmin = max(cDn), min(cDn)
        low_thresh = self.__find_inverse_gmm_cdf(artifact_proportion / 2, gmm, mmax, mmin)
        high_thresh = self.__find_inverse_gmm_cdf(1 - artifact_proportion / 2, gmm, mmax, mmin)
        filtered_mask = np.where((cDn < high_thresh) & (cDn > low_thresh))[0]
        cDn[~filtered_mask] = 0
        return cDn

    
    def denoise(self, signal, artifact_proportion = 0.01):
        thresholded_swt_decomposition = []
        swt_decomposition = pywt.swt(signal, 'Haar')
        for i in range(len(swt_decomposition)):
            cDn = self.__swt_thresholding_denoise(swt_decomposition[i][1], artifact_proportion)
            thresholded_swt_decomposition.append((swt_decomposition[i][0], cDn))
        signal = pywt.iswt(thresholded_swt_decomposition, 'Haar')
        return signal



class EDA_Signal_Processor:


    def eda_clean(self, eda, sampling_rate):
        HIGHCUT_FREQUENCY = 5 # defaults as BioSPPy
        nyquist_freq = 2 * HIGHCUT_FREQUENCY / sampling_rate # Normalize frequency to Nyquist Frequency (Fs/2)
        if 0 < nyquist_freq < 1:
            eda = nk.eda_clean(eda, sampling_rate=sampling_rate, method='biosppy')
        else:
            BUTTERWORTH_ORDER = 4
            CUTOFF_FREQUENCY = 1
            b, a = scipy.signal.butter(BUTTERWORTH_ORDER, CUTOFF_FREQUENCY, btype = 'low', fs = sampling_rate)
            eda = scipy.signal.filtfilt(b, a, eda)
        return eda


    def eda_signal_processing(self, eda, sampling_rate):
        eda_cleaned = self.eda_clean(eda, sampling_rate)
        eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=sampling_rate, method = 'cvxeda')
        scr_peaks, _ = nk.eda_peaks(eda_decomposed['EDA_Phasic'], sampling_rate=sampling_rate)
        signals = pd.DataFrame({"EDA_Cleaned": eda_cleaned})
        signals = pd.concat([signals, eda_decomposed, scr_peaks], axis=1)
        return signals
    

     
    def eda_feature_extraction(self, eda, sampling_rate):
        signals = self.eda_signal_processing(eda, sampling_rate) 
        eda_raw = signals['EDA_Cleaned'].values
        eda_phasic = signals['EDA_Phasic'].values
        eda_tonic = signals['EDA_Tonic'].values

        scr_peaks = signals['SCR_Peaks'].values
        scr_onsets = signals['SCR_Onsets'].values
        scr_amplitude = signals['SCR_Amplitude'].values
        t = np.linspace(0, len(eda_raw), len(eda_raw))

        # Choi J, Ahmed B, Gutierrez-Osuna R. Development and evaluation of an ambulatory stress monitor based on wearable sensors. 
        # IEEE Trans Inf Technol Biomed. 2012 Mar;16(2):279-86. doi: 10.1109/TITB.2011.2169804. Epub 2011 Sep 29. PMID: 21965215.
        mean_scl, std_scr = eda_tonic.mean(), eda_phasic.std()

        # Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, and Kristof Van Laerhoven. 2018. Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.
        # In Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI '18). Association for Computing Machinery, New York, NY, USA, 400–408. DOI:https://doi.org/10.1145/3242969.3242985
        mean_eda, std_eda, min_eda, max_eda = eda_raw.mean(), eda_raw.std(), eda_raw.min(), eda_raw.max()
        eda_dynamic_range = max_eda - min_eda
        mean_scl, std_scl = eda_tonic.mean(), eda_tonic.std()
        num_scr_peaks = scr_peaks.sum()

        res = linregress(t, eda_raw)
        eda_slope = res.slope
        corr, _ = pearsonr(t, eda_tonic)
        
        scr_peaks_index = np.array([index for index in range(len(eda_raw)) if scr_peaks[index] > 0])
        scr_onsets_index = np.array([index for index in range(len(eda_raw)) if scr_onsets[index] > 0])

        if len(scr_onsets_index) > 0 and len(scr_peaks_index) > 0:

            scr_peaks_time = t[scr_peaks_index]
            scr_at_peaks = eda_phasic[scr_peaks_index]
            startle_magnitude = scr_amplitude[scr_peaks_index]

            scr_onsets_time = t[scr_onsets_index]
            scr_at_onsets = eda_phasic[scr_onsets_index]


            if np.isnan(startle_magnitude[0]):
                scr_peaks_index = scr_peaks_index[1:]
                scr_peaks_time = scr_peaks_time[1:]
                startle_magnitude = startle_magnitude[1:]
            elif np.isnan(startle_magnitude[-1]):
                scr_onsets_index = scr_onsets_index[:-1]
                scr_onsets_time = scr_onsets_time[:-1]
                startle_magnitude = startle_magnitude[:-1]

            scr_response_duration = scr_peaks_time - scr_onsets_time
            sum_scr_response_duration = scr_response_duration.sum()
            sum_scr_amplitude = startle_magnitude.sum()
            area_of_response_curve = (scr_response_duration * startle_magnitude).sum() / 2.0
        else:
            scr_at_peaks = np.array([0])
            scr_at_onsets = np.array([0])
            sum_scr_response_duration = 0
            sum_scr_amplitude = 0 
            area_of_response_curve = 0
 

        # Nkurikiyeyezu, K., Yokokubo, A., & Lopez, G. (2019). The Influence of Person-Specific Biometrics in Improving Generic Stress Predictive Models. 
        # ArXiv, abs/1910.01770.
        first_order_grad = np.gradient(eda_phasic)
        second_order_grad = np.gradient(first_order_grad)
        mean_scr, max_scr, min_scr = eda_phasic.mean(), eda_phasic.max(), eda_phasic.min()
        scr_range = max_scr - min_scr
        kurtosis_scr, skewness_scr = kurtosis(eda_phasic), skew(eda_phasic)
        mean_first_grad, std_first_grad = first_order_grad.mean(), first_order_grad.std()
        mean_second_grad, std_second_grad = second_order_grad.mean(), second_order_grad.std()
        mean_peaks, max_peaks, min_peaks, std_peaks = scr_at_peaks.mean(), scr_at_peaks.max(), scr_at_peaks.min(), scr_at_peaks.std()
        mean_onsets, max_onsets, min_onsets, std_onsets = scr_at_onsets.mean(), scr_at_onsets.max(), scr_at_onsets.min(), scr_at_onsets.std()
        ALSC = np.sqrt((np.diff(eda_phasic) ** 2 + 1).sum())
        INSC = np.sum(np.abs(eda_phasic))
        APSC = np.sum(eda_phasic ** 2) / len(eda_phasic)
        RMSC = math.sqrt(APSC)


        statistics_feat = np.array([
                mean_scl, std_scl, std_scr, scr_range, corr, eda_slope, 
                sum_scr_response_duration, sum_scr_amplitude, area_of_response_curve,
                num_scr_peaks, mean_eda, std_eda, min_eda, max_eda, eda_dynamic_range,
                mean_scr, max_scr, min_scr, kurtosis_scr, skewness_scr, mean_first_grad, std_first_grad, mean_second_grad, std_second_grad, 
                mean_peaks, max_peaks, min_peaks, std_peaks, mean_onsets, max_onsets, min_onsets, std_onsets,
                ALSC, INSC, APSC, RMSC])

        return statistics_feat