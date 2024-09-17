#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:17:35 2024

@author: dimitra
"""
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

def train_svm(X_train, Y_train):
    #rbf kernel is used for complex data with no clear decision boundaries
    svm = SVC(kernel='rbf')
    svm.fit(X_train, Y_train)
    return svm

def train_mlp(X_train,Y_train):
    mlp = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    mlp.fit(X_train, Y_train)
    return mlp

def evaluate(X_test,Y_test,classifier,label):
    Y_pred = classifier.predict(X_test)
    # Model Accuracy: how often is the classifier correct?
    print(f'{label} Accuracy:{metrics.accuracy_score(Y_test, Y_pred)}')
    # Model Precision: what percentage of positive tuples are labeled as such?
    print(f'{label} Precision:{metrics.precision_score(Y_test, Y_pred)}')
    # Model Recall: what percentage of positive tuples are labelled as such?
    print(f'{label} Recall:{metrics.recall_score(Y_test, Y_pred)}')
