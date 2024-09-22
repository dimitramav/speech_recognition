# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:29:18 2024

@author: Thanos
"""

import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        # pe = torch.zeros(max_len, d_model)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer('pe', pe)


        # Initialize the position encoding tensor.
        pe = torch.zeros(max_len, d_model)
        
        # Initialize the position tensor.
        # Mind that the unsqueeze function generates a one-dimensional tensor 
        # at the dimension specified by the corresponding input argument.  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Initialize the div_term tensor.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model))
        
        # Replace the even and odd positions of the position encoding.
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        
        # The position encoding tensor should be registered as a buffer that 
        # will not be considered as a model parameter. Buffers, however, are 
        # persistent by default and will be saved alongside parameters. This 
        # behavior may be altered by setting the persistent logical argument to
        # false. Thus, the buffer will not be part of the the module's 
        # state_dict.
        self.register_buffer('pe',pe.unsqueeze(0))
        
    def forward(self, x):
        seq_len= self.pe.size(1)
        
        x = x + self.pe[:seq_len, :]
        return x
    