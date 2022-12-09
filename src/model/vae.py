import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import torch.nn.functional as F

from parsers.midi.midi_features import TOTAL_TICKS
""" Features are (batch_size = 256, 750, 88) 

    Todos: 
        - Might make more sense to pass hidden state to second split, since right now each split is treated as its own piece
"""
class LSTM_Encoder(nn.Module):
    def __init__(self, device, latent_dims, hidden_size, num_layers, dropout):  
        super(LSTM_Encoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.latent_dims = latent_dims
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=88, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.mean = nn.Linear(in_features=hidden_size * TOTAL_TICKS, out_features=latent_dims)
        self.log_var = nn.Linear(in_features=hidden_size * TOTAL_TICKS, out_features=latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0


    def forward(self, x):
        x = x.to(self.device)
        x, hidden = self.lstm(x)

        x = torch.flatten(x, start_dim=1)
        mu = self.mean(x)
        sigma = torch.exp(0.5 * self.log_var(x))

        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        # self.kl = (mu * 0).sum()
        return z  

class LSTM_Decoder(nn.Module):
    
    def __init__(self, device, latent_dims, hidden_size, num_layers, dropout):
        super(LSTM_Decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.latent_dims = latent_dims
        self.num_layers = num_layers

        # self.latent_to_hidden_state = nn.Linear(in_features=latent_dims, out_features=hidden_size)
        # self.latent_to_cell_state = nn.Linear(in_features=latent_dims, out_features=hidden_size)
        self.latent_to_input = nn.Linear(in_features=latent_dims, out_features=hidden_size * TOTAL_TICKS)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.output = nn.Linear(in_features=hidden_size, out_features=88)
        
    def forward(self, x):
        # print(x.shape)
        x = x.to(self.device)
        x = self.latent_to_input(x)
        x = torch.unflatten(x, dim=1, sizes=(TOTAL_TICKS, self.hidden_size))
        # hidden_state = self.latent_to_cell_state(x)
        # cell_state = self.latent_to_cell_state(x)

        # print(lstm_inp.shape, hidden_state.shape, cell_state.shape)
        x, hidden = self.lstm(x)
        x = self.output(x)
        return x

class LSTM_VAE(nn.Module):
    def __init__(self, device, latent_dims, hidden_size, num_layers, dropout=0):
        super(LSTM_VAE, self).__init__()
        self.encoder = LSTM_Encoder(device, latent_dims, hidden_size, num_layers, dropout)
        self.decoder = LSTM_Decoder(device, latent_dims, hidden_size, num_layers, dropout)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        z = self.encoder(x)
        return self.decoder(z)

    def decode(self, z):
        z = z.to(self.device)
        return self.decoder(z)