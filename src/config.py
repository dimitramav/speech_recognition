#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:06:42 2024

@author: dimitra
"""
SAMPLING_RATE = 16000
N_FFT = 1024  
N_MELS = 128  #number of Mel bands
FRAME_LENGTH = 480
HOP_LENGTH = FRAME_LENGTH // 2
FILTER_SIZE = 3
SILENCE_DURATION = 1.0
F_MIN = 50
F_MAX = 500
N_FILES = 5