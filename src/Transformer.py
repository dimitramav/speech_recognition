# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:02:43 2024

@author: Thanos
"""

# Import required libraries.
import torch
import torch.nn as nn
from torch.nn import functional
from src.Transformer_classes.PositionalEncoding import PositionalEncoding
from sklearn.metrics import accuracy_score
import torch.optim as optim
import numpy as np

class Transformer(nn.Module):
    def __init__(self, input_dim,max_len, nhead, num_layers, num_classes):
        # Call the super class constructor.
        super(Transformer, self).__init__()
        # nn.Parameter()
        self.input_dim = input_dim  # Should match n_mels
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes
       
        # Transformer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
               
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(input_dim, max_len=max_len)
        
        self.fc = nn.Linear(input_dim, num_classes)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, src):
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Transformer
        encoder_output = self.transformer_encoder(src)
        
        # Pooling (average over the time dimension)
        pooled_output = encoder_output[-1, :, :]  # Shape: (batch_size, input_dim)            
        
        output = self.fc(pooled_output)
        
        output = self.sigmoid(output)
        
        return output
    
    def train_model(self, data_tensor, label_tensor):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        num_epochs = 10

        # Training loop
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            # Forward pass
            optimizer.zero_grad()
            output = self(data_tensor)  # Transformer expects (seq_len, batch_size, feature_dim)
            
            # labels= functional.one_hot(label_tensor.to(torch.int64),2)
            # Compute loss
            loss = criterion(output, label_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    def evaluate(self, test_data, test_labels, primary_size):
        self.eval()
        
        padded_test_data = self.pad_sequence(test_data, primary_size)
        # padded_test_data = padded_test_data.unsqueeze(0)
        predicted = self(padded_test_data)
        
        truncated_preds = self.truncate_predictions(predicted, target_length=test_data.size(1))        # Get predicted class (index of max logit)
        preds = torch.argmax(truncated_preds, dim=1)
        
        accuracy = accuracy_score(preds, test_labels)       
        
        print(f"Accuracy: {accuracy:.4f}")  
        
    def predict(self, data_tensor, primary_size):
        padded_test_data = self.pad_sequence(data_tensor, primary_size)        
        predictions = self(padded_test_data)
        truncated_preds = self.truncate_predictions(predictions, target_length=data_tensor.size(1))        # Get predicted class (index of max logit)
        predicted_classes = (truncated_preds > 0.5).float()
        return predicted_classes

    def pad_sequence(self,sequence, max_len):
      padding_length = max_len - len(sequence[0])
      return nn.functional.pad(sequence, (0,0,0, padding_length))
  
    
    def truncate_predictions(self,preds, target_length):
        return preds[:target_length]