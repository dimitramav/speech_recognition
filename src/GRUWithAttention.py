# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:48:09 2024

@author: Dell
"""

import torch
import torch.nn as nn
import torch.optim as optim
from src.Transformer_classes.Attention import Attention

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
                     
       context, attn_weights = self.attention(hidden, gru_output)
       
       # Pass through fully connected layer and apply sigmoid for each time step: (1, 3374, 1)
       output = self.fc(context)  # Shape: (1, 3374, 1)
       
       output = self.sigmoid(output)
          
       return output
    
    def train_model(self,dataloader):
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        num_epochs = 6
        for epoch in range(num_epochs):
            self.train()
            
            for data, labels in dataloader:
                output = self(data)
            
                loss = criterion(output, labels)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    def evaluate(self, dataloader):
        self.eval()
        
        total_correct = 0
        for test_data, test_labels in dataloader:
            predicted = self(test_data)
            predicted_classes = (predicted > 0.5).float()
            correct_predictions = (predicted_classes == test_labels).float()
            total_correct += correct_predictions.sum().item()
        accuracy = total_correct / dataloader.dataset.labels.size(0)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        
        
    def predict(self, data_tensor):   
        all_predicted_classes = []
        
        for test_data in data_tensor:
            predictions = self(test_data)
            predicted_classes=(predictions > 0.5).float()
            
            # Append the predicted classes to the list
            all_predicted_classes.append(predicted_classes)
            
        all_predicted_classes = torch.cat(all_predicted_classes, dim=0)
        return predicted_classes