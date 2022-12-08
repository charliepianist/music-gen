from parsers.main import save_data
from model.vae import VariationalAutoencoder
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parsers.common import get_data_dir, get_model_dir
from model.df_dataset import DfDataset
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import DatasetFolder
from parsers.midi.midi_features import TOTAL_TICKS, NUM_NOTES, pd_row_to_torch, save_torch_to_midi
from train import DATA_DIR_DATASET_FOLDER, TRAIN_VALID_SPLIT, LATENT_DIMS, MANUAL_SEED, DATA_DIR

OUT_FILE = get_model_dir('eval.mid')

if __name__ == '__main__':
    reconstruct_thresh = float(input('A note must be >= to start ([0,1]): '))
    reconstruct_contig_thresh = float(input('A note must be >= to continue ([0,1]): '))
    torch.manual_seed(MANUAL_SEED)
    def getitem(file):
        row = pd.read_pickle(file)
        x = pd_row_to_torch(row)
        print(row[[column for column in row.index if 'feature_' not in column]])
        return torch.tensor(x)
    
    # Copied from train
    inp = input('Was it trained with all data in memory? (Y/N, default N)')
    cache_all = False
    if inp == 'Y' or inp == 'y':
        cache_all = True
        modulo = int(input('Modulo for data (1 to use all):'))
    if cache_all: 
        dataset = DfDataset(DATA_DIR, modulo=modulo)
    else:
        dataset = DatasetFolder(DATA_DIR_DATASET_FOLDER, getitem, extensions=('pkl',))
    m = len(dataset)
    train_data, val_data = random_split(dataset, [m - int(m*TRAIN_VALID_SPLIT), int(m*TRAIN_VALID_SPLIT)])
    train_loader = DataLoader(train_data, batch_size=1)
    valid_loader = DataLoader(val_data, batch_size=1)
    loader = valid_loader
    inp = input('Use in sample data? (Y/y for yes)')
    if inp == 'Y' or inp == 'y':
        loader = train_loader

    # Parameterization
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    vae = VariationalAutoencoder(device, latent_dims=LATENT_DIMS)
    vae.load_state_dict(torch.load(get_model_dir('vae_checkpoint_' + input('Model #: ') + '.pth')))
    vae.eval()
    if torch.cuda.is_available():
        vae.cuda()

    # Compute features
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x = x.view(-1, 1, TOTAL_TICKS, NUM_NOTES)
            encoded_data = vae.encoder(x)
            reconstructed = vae.decode(encoded_data)
            # Returns (1, 1, *features) shape (since first two are batch_size, num_channels)

            # Convert to midi and save
            # for i in range(len(reconstructed[0][0])):
            #     for j in range(len(reconstructed[0][0][0])):
            #         if reconstructed[0][0][i][j] != 0:
            #             print(reconstructed[0][0][i][j])
            save_torch_to_midi(reconstructed[0][0].cpu(), OUT_FILE, reconstruct_thresh=reconstruct_thresh, reconstruct_contig_thresh=reconstruct_contig_thresh)
            break