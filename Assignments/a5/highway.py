#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1h

class Highway(nn.Module):
    """ Highway network for ConvNN
        - Relu
        - Sigmoid
        - gating mechanism from LSTM
    """

    def __init__(self,embed_size):
        """ Init Higway network
            @param embed_size (int): Embedding size of word, in handout, 
                                     it's e_{word} (dimensionality)
        """

        super(Highway, self).__init__()

        self.projection = nn.Linear(embed_size,embed_size,bias = True)
        self.gate = nn.Linear(embed_size,embed_size, bias = True)
        

    def forward(self, X_conv_out):
        """
            Take mini-batch of sentence of ConvNN
            @param X_conv_out (Tensor): Tensor with shape (max_sentence_length, batch_size, embed_size)
            @return X_highway (Tensor): combinded output with shape (max_sentence_length, batch_size, embed_size)
        """
        
        X_proj = F.relu(self.projection(X_conv_out))
        X_gate = torch.sigmoid(self.gate(X_conv_out))
        X_highway =  torch.mul(X_gate, X_proj) + torch.mul((1 - X_gate),X_conv_out)

        return X_highway


### END YOUR CODE 

