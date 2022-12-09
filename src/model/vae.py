import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import torch.nn.functional as F

# Features are TOTAL_TICKS x NUM_NOTES
from parsers.midi.midi_features import TOTAL_TICKS, NUM_NOTES

class VariationalEncoder(nn.Module):
    def __init__(self, device, latent_dims):  
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (5, 1), stride=(2,1), padding=(2,0)) # Very near in time
        self.conv2 = nn.Conv2d(16, 32, (101, 1), stride=(2,1), padding=(50,0)) # around 8 seconds wide
        self.batch2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 88), stride=(2,1), padding=(2,0)) # every 8 second window for ~1 second
        # self.batch3 = nn.BatchNorm2d(32)
        # self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=0)  
        # self.linear1 = nn.Linear(64 * 45 * 3, 128)
        self.linear1 = nn.Linear(64 * 94 * 1, 256)
        self.linear2 = nn.Linear(256, latent_dims)
        self.linear3 = nn.Linear(256, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = F.leaky_relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        # x = F.relu(self.batch3(self.conv3(x)))
        # x = F.relu(self.conv4(x))
        x = F.leaky_relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        # self.kl = (mu * 0).sum()
        return z  

class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 64 * 94 * 1),
            nn.LeakyReLU(True)
        )

        # self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 45, 3))
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 94, 1))

        self.decoder_conv = nn.Sequential(
            # nn.ConvTranspose2d(64, 32, 3, stride=2, padding=0, output_padding=(0,1)),
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, (5, 88), stride=(2,1), padding=(2,0), output_padding=(1,0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 16, (101, 1), stride=(2,1), padding=(50,0), output_padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16, 1, (5,1), stride=(2,1), padding=(2,0), output_padding=(1,0))
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
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