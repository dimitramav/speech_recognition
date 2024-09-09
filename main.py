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
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

# Relative path to the folder
folder_path = Path('data/CleanSpeech_training/')

file_count=0
# List files in the folder
for file_path in folder_path.iterdir():
    if file_path.is_file() and file_path.suffix == '.wav':
        y, sr = librosa.load(file_path)
        # Convert the waveform to a Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        
        # Convert the Mel spectrogram to a decibel scale
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Plot the Mel spectrogram
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram (dB)')
        plt.tight_layout()
        plt.show()
        # Increment the file count and break if five files have been read
        file_count += 1
        if file_count >= 5:
            break
