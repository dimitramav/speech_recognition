#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:36:58 2024

@author: dimitra
"""
import librosa
import librosa.display
import numpy as np
from src.config import SAMPLING_RATE,N_FFT,HOP_LENGTH,N_MELS


def convert_to_mel(audio):
    # Convert the waveform to a Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    # Transpose the spectrogram to have the shape (timesteps, n_mels)
    # Rows now represent time frames
    # Columns represent Mel frequency bins
    mel_spectrogram = mel_spectrogram.T 
    return mel_spectrogram
    
def create_dataset(folder_path,label):
    file_count=0
    # List files in the folder
    audio_time_fragments = []
    target_labels = []
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix == '.wav':
            audio,sampling_rate = librosa.load(file_path, sr=SAMPLING_RATE)
            mel_spectrogram = convert_to_mel(audio)
            audio_time_fragments.append(mel_spectrogram)
            target_labels.append(label)
            # Increment the file count and break if five files have been read
            file_count += 1
            if file_count >= 5:
                break
    # Combine time_fragments into single matrix along the time axis. Shape of audio data is (n_time_frames, n_mel_bins=128)
    audio_data = np.vstack(audio_time_fragments)
    #  Create array of labels, where each time frame (or element in the stacked Mel spectrogram) is labeled as 1
    labels = np.ones(audio_data.shape[0]) if label==1 else np.zeros(audio_data.shape[0])
    return audio_data,labels

def combine_datasets(X1,Y1,X2,Y2):
    # 2D arrays concatenation
    X = np.vstack((X1,X2))
    # 1D arrays concatenation
    Y = np.hstack((Y1,Y2))
    return X, Y
