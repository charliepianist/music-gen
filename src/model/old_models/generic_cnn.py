import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import torch.nn.functional as F

"""
I tried this with 20 latent Dims, trained for 10 epochs and stopped improving, was quite bad.
"""

# Features are TOTAL_TICKS x NUM_NOTES
from parsers.midi.midi_features import TOTAL_TICKS, NUM_NOTES

class VariationalEncoder(nn.Module):
    def __init__(self, device, latent_dims):  
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 7, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 7, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 7, stride=2, padding=1)
        # self.batch3 = nn.BatchNorm2d(32)
        # self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=0)  
        # self.linear1 = nn.Linear(64 * 45 * 3, 128)
        self.linear1 = nn.Linear(32 * 91 * 8, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        # x = F.relu(self.batch3(self.conv3(x)))
        # x = F.relu(self.conv4(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        # self.kl = (mu * 0).sum()
        return z  

class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super().__init__()

        # self.decoder_lin = nn.Sequential(
        #     nn.Linear(latent_dims, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 64 * 45 * 3),
        #     nn.ReLU(True)
        # )
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 32 * 91 * 8),
            nn.ReLU(True)
        )

        # self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 45, 3))
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 91, 8))

        self.decoder_conv = nn.Sequential(
            # nn.ConvTranspose2d(64, 32, 3, stride=2, padding=0, output_padding=(0,1)),
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 7, stride=2, padding=1, output_padding=(0,0)),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 7, stride=2, padding=1, output_padding=(0,1)),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 7, stride=2, padding=1, output_padding=(1,1))
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, device, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(device, latent_dims)
        self.decoder = Decoder(latent_dims)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        z = self.encoder(x)
        return self.decoder(z)

    def decode(self, z):
        z = z.to(self.device)
        return self.decoder(z)