# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:53:19 2024

@author: Thanos
"""
import torch
import torch.nn as nn
import torch.optim as optim

class GRUclassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUclassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(1), self.gru.hidden_size)
        out, _ = self.gru(x, h0)  # out: tensor (batch, seq_len, hidden_size)
        out = out[-1, :, :]  # Κρατάμε την τελευταία χρονική στιγμή
        out = self.fc(out)   # out: tensor (batch, output_size)
        out = self.sigmoid(out)
        return out
    
    def train_model(self,data_tensor, label_tensor):
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        num_epochs = 150
        for epoch in range(num_epochs):
            self.train()     
            
            outputs = self(data_tensor)
            loss = criterion(outputs, label_tensor)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    def evaluate(self, X_test, Y_test):
        self.eval()
        test_data=torch.from_numpy(X_test).float()
        test_data = test_data.unsqueeze(0)  # Dummy test data
        test_labels=torch.from_numpy(Y_test).float()
        test_labels=test_labels.unsqueeze(1)
        
        predicted = self(test_data)
        predicted_classes = (predicted > 0.5).float()
        correct_predictions = (predicted_classes == test_labels).float()
        accuracy = correct_predictions.sum() / test_labels.size(0)
        print(f'RNN Accuracy: {accuracy.item() * 100:.2f}%')
        
        
    def predict(self, data_tensor):     
        predictions = self(data_tensor)
        predicted_classes = (predictions > 0.5).float()
        return predicted_classes