#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:06:37 2024

@author: dimitra

@article{reddy2019scalable,
  title={A Scalable Noisy Speech Dataset and Online Subjective Test Framework},
  author={Reddy, Chandan KA and Beyrami, Ebrahim and Pool, Jamie and Cutler, Ross and Srinivasan, Sriram and Gehrke, Johannes},
  journal={Proc. Interspeech 2019},
  pages={1816--1820},
  year={2019}
}
"""

from pathlib import Path
import librosa
import librosa.display
import numpy as np
from sklearn.model_selection import train_test_split

SAMPLING_RATE = 16000
N_FFT = 2048  #number of windows
HOP_LENGTH = N_FFT//2 #step size
N_MELS = 128  #number of Mel bands

# Relative path to the folder
folder_path = Path('data/CleanSpeech_training/')

file_count=0
# List files in the folder
audio_data = []
target_labels = []
for file_path in folder_path.iterdir():
    if file_path.is_file() and file_path.suffix == '.wav':
        array,sampling_rate = librosa.load(file_path, sr=SAMPLING_RATE)
        # Convert the waveform to a Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=array, sr=SAMPLING_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        # Transpose the spectrogram to have the shape (timesteps, n_mels)
        mel_spectrogram = mel_spectrogram.T
        audio_data.append(mel_spectrogram)
        target_labels.append(1)
        # Increment the file count and break if five files have been read
        file_count += 1
        if file_count >= 5:
            break

X_train, X_test, y_train, y_test = train_test_split(audio_data, target_labels, test_size=0.2, random_state=42)
# Ensure all spectrograms have the same shape
#max_length = max([spec.shape[0] for spec in audio_data])
#X_train = [np.pad(spec, ((0, max_length - spec.shape[0]), (0, 0)), mode='constant') for spec in X_train]
#X_test = [np.pad(spec, ((0, max_length - spec.shape[0]), (0, 0)), mode='constant') for spec in X_test]
 
# Convert to NumPy arrays
#X_train = np.array(X_train)
#X_test = np.array(X_test)

print(X_train)