""" This function returns the specified number of mel filters. """

import numpy as np

def mel_filter_make(nfilt, nfft = 512):
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (16000 / 2) / 700)) 
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  
    hz_points = (700 * (10**(mel_points / 2595) - 1)) 
    bin = np.floor((nfft) * hz_points / 16000)

    fbank = np.zeros((nfilt, int(np.floor(nfft))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   
        f_m = int(bin[m])             
        f_m_plus = int(bin[m + 1])    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    return fbank