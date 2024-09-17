#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:10:35 2024

@author: dimitra
"""
import numpy as np
from src.config import FRAME_LENGTH,HOP_LENGTH,SAMPLING_RATE

def extract_clean_speech_intervals(audio, predictions):
    # Initialize an array for the reconstructed audio
    speech_intervals = []
    start = None
    for i, pred in enumerate(predictions):
        if pred == 1 and start is None:
            # Sampling_rate division converts index to time
            start = i * HOP_LENGTH / SAMPLING_RATE
        elif pred == 0 and start is not None:
            end = i * HOP_LENGTH / SAMPLING_RATE
            speech_intervals.append((start, end))
            start = None
    
    # Speech is the last predicted element of the array
    if start is not None:
        end = len(predictions) * HOP_LENGTH / SAMPLING_RATE
        speech_intervals.append((start, end))    
    return speech_intervals
    
def extract_clean_speech(intervals,audio):
    extracted_speech = np.array([])
    # Loop through each interval and extract the corresponding audio samples
    for start_time, end_time in intervals:
         # Convert time to sample index
        start_sample = int(start_time * SAMPLING_RATE) 
        end_sample = int(end_time * SAMPLING_RATE) 
        segment = audio[start_sample:end_sample]
        extracted_speech = np.concatenate((extracted_speech, segment))
    return(extracted_speech)