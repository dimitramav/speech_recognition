#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:17:35 2024

@author: dimitra
"""
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
#from src.GRUclassifier import GRUclassifier
#import torch
#import torch.nn as nn
#import torch.optim as optim
import numpy as np

def train_svm(X_train, Y_train):
    #rbf kernel is used for complex data with no clear decision boundaries
    svm = SVC(kernel='rbf')
    svm.fit(X_train, Y_train)
    return svm

def train_mlp(X_train,Y_train):
    mlp = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    mlp.fit(X_train, Y_train)
    return mlp
"""
def train_rnn(X_train,Y_train):
    #batching my data to one single batch
    data_tensor = torch.from_numpy(X_train).float()
    data_tensor = data_tensor.unsqueeze(0)
    label_tensor = torch.from_numpy(Y_train).float()
    label_tensor = label_tensor.unsqueeze(1)    
    
    input_size = data_tensor.size(2) #every element has 128 NMels and thats the input size 
    hidden_size = 64 #Hyperparameter   
    output_size = 1 #output for each element
    num_layers = 2  #Hyperparameter
    
    rnn= GRUclassifier(input_size, hidden_size, output_size, num_layers)
    
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(rnn.parameters(), lr=0.001)
    
    num_epochs = 150
    for epoch in range(num_epochs):
        rnn.train()     
        
        outputs = rnn(data_tensor)
        loss = criterion(outputs, label_tensor)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return rnn

def rnn_evaluate(X_test,Y_test, rnn):
    rnn.eval()
    test_data=torch.from_numpy(X_test).float()
    test_data = test_data.unsqueeze(0)  # Dummy test data
    test_labels=torch.from_numpy(Y_test).float()
    test_labels=test_labels.unsqueeze(1)
    
    predicted = rnn(test_data)
    predicted_classes = (predicted > 0.5).float()
    correct_predictions = (predicted_classes == test_labels).float()
    accuracy = correct_predictions.sum() / test_labels.size(0)
    print(f'RNN Accuracy: {accuracy.item() * 100:.2f}%')
"""

def evaluate(X_test,Y_test,classifier,label):
    Y_pred = classifier.predict(X_test)
    # Model Accuracy: how often is the classifier correct?
    print(f'{label} Accuracy:{metrics.accuracy_score(Y_test, Y_pred)}')
    # Model Precision: what percentage of positive tuples are labeled as such?
    print(f'{label} Precision:{metrics.precision_score(Y_test, Y_pred)}')
    # Model Recall: what percentage of positive tuples are labelled as such?
    print(f'{label} Recall:{metrics.recall_score(Y_test, Y_pred)}')
