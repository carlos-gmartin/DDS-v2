# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:16:41 2023

@author: Carlos G Martin
"""

# Version 1.2
# Currently used for processing a magnitude spectrum into a triangular_filterbank which outputs features.
# Need to make an improved comment for the function.
# The parameters are incredibly important, need to output graphs for reasoning.
# Current working model.

import numpy as np
import matplotlib.pyplot as plt


# Convert to melScale
def toMelScale(f):
    melF = 1125*np.log(1 + f/700)
    return melF

# Convert Back to frequency
def toFrequencyScale(melF):
    f = 700 *(np.exp(melF/1125)- 1)
    return f


# Function to compute filterbank features.
def triangular_filterbank_features(magnitude_spectrum, frameLength, banks=120, sample_rate=16000, min_freq=0, max_freq=8000):
    # # Calculate the Mel scale minimum and maximum frequencies
    mel_min = toMelScale(min_freq)
    mel_max = toMelScale(max_freq)
    
    # # Calculate Mel center frequencies, evenly spaced within the Mel scale range
    mel_center_frequencies = np.linspace(mel_min, mel_max, num=banks)
    
    # Convert Mel center frequencies back to linear frequencies
    center_frequencies = toFrequencyScale(mel_center_frequencies)
   
    
    # Initialize an array to store the computed filterbank features
    filterbank_features = np.zeros(banks)
    
    #Min Max scaler. Converting frequency points to bin points (magSpec indices)
    binPoints = (center_frequencies - min_freq) / (max_freq - min_freq)
    binPoint = np.floor(binPoints * (frameLength - 0) + 0)
    
    # Initialize an array to represent the filter shape
    filter_shape = np.zeros((frameLength, banks-2))

    eps = 1**(-16)
    for i in range(1,banks-1):
        for k in range(0,frameLength):
            # Check if frequency bin k is within the range of the current filterbank
            if k < binPoint[i-1]:
                filter_shape[k][i-1] = 0
            elif k >= binPoint[i - 1] and k <= binPoint[i]:
                filter_shape[k][i - 1] = (k - binPoint[i - 1]) / ((binPoint[i] - binPoint[i - 1]) + eps)
            elif k >= binPoint[i] and k <= binPoint[i + 1]:
                filter_shape[k][i - 1] = (binPoint[i + 1] - k) / ((binPoint[i + 1] - binPoint[i]) + eps)
            elif k > binPoint[i + 1]:
                filter_shape[k][i - 1] = 0

    # Compute the filterbank feature for each frame by multiplying the magnitude spectrum
    filterbank_features = np.matmul(filter_shape.T, magnitude_spectrum)
    
    # plt.figure()
    # plt.plot(filterbank_features)
    # plt.show()
    
    return filterbank_features