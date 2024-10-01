# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:48:09 2024

@author: Dell
"""

import torch
import torch.nn as nn
import torch.optim as optim
from src.Transformer_classes.Attention import Attention
import time

class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, dropout=0.1):
        super(GRUWithAttention, self).__init__()
        # GRU layer
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        
        # Attention layer (this will output attention weights)
        self.attention = Attention(hidden_size, batch_size)
                
        # Fully connected layer for each time step
        self.fc = nn.Linear(input_size * 2, output_size)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       gru_output, hidden = self.gru(x)       
       # outputs = [batch_size, src_len, enc_hid_dim * 2]
       # hidden = [n_layers * num_directions, batch_size, enc_hid_dim]  
         
       context, attn_weights = self.attention(hidden, gru_output)
       
       # Pass through fully connected layer and apply sigmoid
       output = self.fc(context)  # Shape: (batch_size, output)
       
       output = self.sigmoid(output) #normalize between 0 and 1
          
       return output
    
    def train_model(self,dataloader):
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        num_epochs = 6
        for epoch in range(num_epochs):
            self.train()
            
            for data, labels in dataloader: #iterating in every batch of data and labels with N-data= batch_size
                output = self(data)
            
                loss = criterion(output, labels)
        
                optimizer.zero_grad() 
                loss.backward()   #applying backward propagation in every batch of data
                optimizer.step()
                
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    def evaluate(self, dataloader):
        start_time = time.time()

        self.eval()
        
        total_correct = 0
        for test_data, test_labels in dataloader:
            predicted = self(test_data)
            predicted_classes = (predicted > 0.5).float()
            correct_predictions = (predicted_classes == test_labels).float()
            total_correct += correct_predictions.sum().item()
            
        accuracy = total_correct / dataloader.dataset.labels.size(0)
        end_time = time.time()
        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f'Evaluation Time: {end_time-start_time:.4f} seconds')
        
    def predict(self, data_tensor):   
        all_predicted_classes = []
        
        for test_data in data_tensor:
            predictions = self(test_data)
            predicted_classes=(predictions > 0.5).float()
            
            # Append the predicted classes to the list
            all_predicted_classes.append(predicted_classes)
            
        all_predicted_classes = torch.cat(all_predicted_classes, dim=0)
        return all_predicted_classes