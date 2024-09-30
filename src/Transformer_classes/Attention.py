# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:03:56 2024

@author: Dell
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size, batch_size):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size * 2, hidden_size)
        self.V = nn.Linear(hidden_size, batch_size)

    def forward(self, query, keys):
        # hidden: (batch_size, seq_len, hidden_size * 2)
        # encoder_outputs: (batch_size, seq_len, hidden_size * 2)
        
        # Apply attention mechanism
        scores = self.V(torch.tanh(self.W1(query) + self.W2(keys)))
        
        # Normalize energies to obtain weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # Compute context vector as weighted sum of encoder outputs for each time step
        context = torch.mm(attn_weights, keys)  # (batch_size, seq_len, hidden_size * 2)
        
        return context, attn_weights