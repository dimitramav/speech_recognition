# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:53:19 2024

@author: Thanos
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time

class GRUclassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUclassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(1), self.gru.hidden_size)
        out, _ = self.gru(x, h0)  # out: tensor (batch, seq_len, hidden_size)
        out = out[-1, :, :]  # Keeping the last timestamp
        out = self.fc(out)   # out: tensor (n_samples, output_size)
        out = self.sigmoid(out) #normalize the outputs between 0 and 1
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
            loss.backward() #Backward Propagation over time
            optimizer.step()

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    def evaluate(self, X_test, Y_test):
        start_time = time.time()

        self.eval()
        #Transforming the test data
        test_data=torch.from_numpy(X_test).float()
        test_data = test_data.unsqueeze(0)  # Dummy test data
        test_labels=torch.from_numpy(Y_test).float()
        test_labels=test_labels.unsqueeze(1)
        
        predicted = self(test_data)
        end_time = time.time()
        predicted_classes = (predicted > 0.5).float() #Normalize the predictions to 0=noise or 1=speech
        correct_predictions = (predicted_classes == test_labels).float()
        accuracy = correct_predictions.sum() / test_labels.size(0)
        print(f'GRU Accuracy: {accuracy.item() * 100:.2f}%')
        print(f'GRU Evaluation Time: {end_time-start_time:.4f} seconds')

        
        
    def predict(self, data_tensor):     
        predictions = self(data_tensor)
        predicted_classes = (predictions > 0.5).float()
        return predicted_classes