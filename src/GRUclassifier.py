# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:53:19 2024

@author: Thanos
"""
import torch
import torch.nn as nn

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