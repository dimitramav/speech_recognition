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
from src.classifiers import train_svm,train_mlp,evaluate,train_rnn, train_gru_with_attention, to_tensor
from sklearn.model_selection import train_test_split
from scipy.signal import medfilt
from src.config import SAMPLING_RATE,FILTER_SIZE
from src.audio import extract_clean_speech_intervals,extract_clean_speech
from src.AudioDataset import AudioDataset
from torch.utils.data import DataLoader
import librosa
import soundfile as sf
import torch


if __name__=="__main__":
    
    #Preprocessing
    X_clean_data, Y_clean_data = create_dataset(Path('data/CleanSpeech_training/'),1) #clean speech data with label 1
    X_noise_data, Y_noise_data = create_dataset(Path('data/Noise_training/'),0) #noise data with label 0
    X,Y = combine_datasets(X_clean_data, Y_clean_data, X_noise_data, Y_noise_data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("Επιλέξτε μοντέλο: ")
    print("SVM: 1, MLP:2 GRU:3 Attention Mechanism:4")

    model= input()
    
    #Noisy Speech data for prediction
    audio,_ = librosa.load('data/NoisySpeech_training/noisy11_SNRdb_0.0_clnsp11.wav', sr=SAMPLING_RATE)  #Noisy speech file loading        
    mel_noisyspeech = convert_to_mel(audio) # conversion to mel spectogram
    classifier_name = ""
    if(model =='1'):
        #SVM  
        classifier_name="svm"
        svm = train_svm(X_train, Y_train)
        evaluate( X_test, Y_test,svm,"SVM") 
               
        predictions = svm.predict(mel_noisyspeech) 
        
    if(model == '2'):
        # #MLP
        classifier_name="mlp"
        mlp = train_mlp(X_train, Y_train)
        evaluate(X_test,Y_test,mlp,"MLP")
               
        predictions = mlp.predict(mel_noisyspeech)    
    
    if(model == '3'):
        #RNN
        classifier_name="gru"
        rnn = train_rnn(X_train, Y_train)
        rnn.evaluate(X_test,Y_test)
                
        data_tensor = torch.from_numpy(mel_noisyspeech).float() #convert to float tensor [n_samples,n_mels]
        data_tensor = data_tensor.unsqueeze(0) #[1,n_samples, n_mels] (single batch)
        predictions= rnn.predict(data_tensor)   
        predictions= predictions.squeeze(1).numpy() #prepare for medfilt which accepts nparray of 1 dimension

    if(model == '4'):
        #Attention Mechanism
        classifier_name = "attention"
        attention = train_gru_with_attention(X_train, Y_train)
        test_dataset = AudioDataset(X_test, Y_test) 
        dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)

        attention.evaluate(dataloader)
                
        data_tensor = torch.from_numpy(mel_noisyspeech).float()
        predict_loader= DataLoader(data_tensor, batch_size=4, shuffle=True, drop_last=True)
        predictions= attention.predict(predict_loader)
        predictions= predictions.squeeze(1).numpy()


    predictions = medfilt(predictions, kernel_size=FILTER_SIZE)  #apply median filtering to predictions         
    cleanspeech_intervals = extract_clean_speech_intervals(audio,predictions)
    cleanspeech, fundamental_frequency = extract_clean_speech(cleanspeech_intervals,audio)
    print(f'Fundamental Frequency: {fundamental_frequency} Hz')
    sf.write(f'reconstructed_speech_{classifier_name}.wav', cleanspeech, SAMPLING_RATE)
    

