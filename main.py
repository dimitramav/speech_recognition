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
from src.data_manipulation import create_dataset,combine_datasets
from src.classifiers import train_svm,train_mlp,evaluate
from sklearn.model_selection import train_test_split


if __name__=="__main__":
    X_clean_data, Y_clean_data = create_dataset(Path('data/CleanSpeech_training/'),1)
    X_noise_data, Y_noise_data = create_dataset(Path('data/Noise_training/'),0)
    X,Y = combine_datasets(X_clean_data, Y_clean_data, X_noise_data, Y_noise_data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    #SVM
    svm = train_svm(X_train, Y_train)
    evaluate( X_test, Y_test,svm,"SVM")
        
    #MLP
    mlp = train_mlp(X_train, Y_train)
    evaluate(X_test,Y_test,mlp,"MLP")
    

