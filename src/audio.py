#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:10:35 2024

@author: dimitra
"""
import numpy as np
from src.config import FRAME_LENGTH,HOP_LENGTH,SAMPLING_RATE,SILENCE_DURATION,F_MIN,F_MAX
import librosa

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
    silence = np.zeros(int(SAMPLING_RATE * SILENCE_DURATION))
    speaker_frequencies = []
    # Loop through each interval and extract the corresponding audio samples
    for start_time, end_time in intervals:
         # Convert time to sample index
        start_sample = int(start_time * SAMPLING_RATE) 
        end_sample = int(end_time * SAMPLING_RATE) 
        segment = audio[start_sample:end_sample]
        f0 = calculate_speaker_frequency(segment)
        speaker_frequencies.append(f0)
        extracted_speech = np.concatenate((extracted_speech, segment))
        extracted_speech = np.concatenate((extracted_speech, silence))
    return extracted_speech, np.mean(speaker_frequencies)

def calculate_speaker_frequency(segment):
    pitches = extract_pitches(segment)
    # Remove zero values (pitches corresponding to silence)
    non_zero_pitches = pitches[pitches > 0]
    # Return the mean of the non-zero pitches if any, else return 0
    if len(non_zero_pitches) > 0:
        return np.mean(non_zero_pitches)
    else:
        return 0
        

def extract_pitches(y):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=SAMPLING_RATE, fmin=F_MIN, fmax=F_MAX)
    # get indexes of the maximum value in each time slice
    max_indexes = np.argmax(magnitudes, axis=0)
    # get the pitches of the max indexes per time slice
    pitches = pitches[max_indexes, range(magnitudes.shape[1])]
    return pitches