"""
!/usr/bin/env python3
-*- coding: utf-8 -*-
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
from src.data_manipulation import create_dataset,combine_datasets,convert_to_mel
from src.classifiers import train_svm,train_mlp,evaluate,train_rnn, train_transformer, to_tensor
from sklearn.model_selection import train_test_split
from scipy.signal import medfilt
from src.config import SAMPLING_RATE,FILTER_SIZE
from src.audio import extract_clean_speech_intervals,extract_clean_speech
import librosa
import soundfile as sf
import torch


if __name__=="__main__":
    
    #Preprocessing
    X_clean_data, Y_clean_data = create_dataset(Path('data/CleanSpeech_training/'),1)
    X_noise_data, Y_noise_data = create_dataset(Path('data/Noise_training/'),0)
    X,Y = combine_datasets(X_clean_data, Y_clean_data, X_noise_data, Y_noise_data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("Επιλέξτε μοντέλο: ")
    print("SVM: 1, MLP:2 RNN:3 Transformer:4")

    model= input()
    
    #Noisy Speech data for prediction
    audio,_ = librosa.load('data/NoisySpeech_training/noisy2_SNRdb_0.0_clnsp2.wav', sr=SAMPLING_RATE)        
    mel_noisyspeech = convert_to_mel(audio)
    
    if(model =='1'):
        #SVM        
        svm = train_svm(X_train, Y_train) #training
        evaluate( X_test, Y_test,svm,"SVM") #evaluation
               
        predictions = svm.predict(mel_noisyspeech) #prediction in noizy speech  
        predictions = medfilt(predictions, kernel_size=FILTER_SIZE)        
        
    if(model == '2'):
        # #MLP
        mlp = train_mlp(X_train, Y_train)
        evaluate(X_test,Y_test,mlp,"MLP")
               
        predictions = mlp.predict(mel_noisyspeech)    
        predictions = medfilt(predictions, kernel_size=FILTER_SIZE)        
    
    if(model == '3'):
        #RNN
        rnn = train_rnn(X_train, Y_train)
        rnn.evaluate(X_test,Y_test)
                
        data_tensor = torch.from_numpy(mel_noisyspeech).float()
        data_tensor = data_tensor.unsqueeze(0)        
        predictions= rnn.predict(data_tensor)        

    if(model == '4'):
        #Transformer
        transformer = train_transformer(X_train, Y_train)
        test_data,test_labels= to_tensor(X_test, Y_test)
        transformer.evaluate(test_data, test_labels, X_train.shape[0])
                
        data_tensor = torch.from_numpy(mel_noisyspeech).float()
        data_tensor = data_tensor.unsqueeze(0)        
        predictions= transformer.predict(data_tensor, X_train.shape[0])
        
    cleanspeech_intervals = extract_clean_speech_intervals(audio,predictions)
    cleanspeech, fundamental_frequency = extract_clean_speech(cleanspeech_intervals,audio)
    print(f'Fundamental Frequency: {fundamental_frequency} Hz')
    sf.write('reconstructed_speech_transformer.wav', cleanspeech, SAMPLING_RATE)
    

