#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:17:35 2024

@author: dimitra
"""
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import time
from src.GRUclassifier import GRUclassifier
from src.Transformer import Transformer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def train_svm(X_train, Y_train):
    #rbf kernel is used for complex data with no clear decision boundaries
    svm = SVC(kernel='rbf', gamma = 1)
    svm.fit(X_train, Y_train)
    return svm

def train_mlp(X_train,Y_train):
    mlp = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    mlp.fit(X_train, Y_train)
    return mlp

def train_rnn(X_train,Y_train):
    data_tensor, label_tensor = to_tensor(X_train, Y_train)
    
    input_size = data_tensor.size(2) #every element has 128 NMels and thats the input size 
    hidden_size = 64 #Hyperparameter   
    output_size = 1 #output for each element
    num_layers = 2  #Hyperparameter
    
    rnn= GRUclassifier(input_size, hidden_size, output_size, num_layers)
    rnn.train_model(data_tensor, label_tensor)
    
    return rnn

def train_transformer(X_train,Y_train):    
    data_tensor, label_tensor = to_tensor(X_train, Y_train)    
    
    input_size = data_tensor.size(2) #every element has 128 NMels and thats the input size 
    feature_dim= data_tensor.size(1)
    
    transformer = Transformer(input_dim=input_size, max_len=feature_dim, nhead=8, num_layers=4, num_classes=1)    
    
    transformer.train_model(data_tensor, label_tensor)
    
    return transformer
    
def evaluate(X_test,Y_test,classifier,label):
    start_time = time.time()
    Y_pred = classifier.predict(X_test)
    end_time = time.time()
    # Model Accuracy: how often is the classifier correct?
    print(f'{label} Accuracy:{metrics.accuracy_score(Y_test, Y_pred)}')
    # Model Precision: what percentage of positive tuples are labeled as such?
    print(f'{label} Precision:{metrics.precision_score(Y_test, Y_pred)}')
    # Model Recall: what percentage of positive tuples are labelled as such?
    print(f'{label} Recall:{metrics.recall_score(Y_test, Y_pred)}')
    print(f'{label} Evaluation Time: {end_time-start_time:.4f} seconds')


def to_tensor(X_train, Y_train):
    #batching my data to one single batch
    data_tensor = torch.from_numpy(X_train).float()
    data_tensor = data_tensor.unsqueeze(0)
    label_tensor = torch.from_numpy(Y_train).float()
    label_tensor = label_tensor.unsqueeze(1)    
    return data_tensor, label_tensor
