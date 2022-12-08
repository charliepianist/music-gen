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
from torch.nn import BCELoss
from parsers.midi.midi_features import TOTAL_TICKS, NUM_NOTES, pd_row_to_torch

DATA_DIR = get_data_dir('rows', 'class_0')
DATA_DIR_DATASET_FOLDER = get_data_dir('rows')
DATA_MINIMAL_DIR = get_data_dir('rows', 'minimal')
def get_model_checkpoint_file(epochs):
    return get_model_dir('vae_checkpoint_' + str(epochs) + '.pth')
def get_loss_checkpoint_file(epochs):
    return get_model_dir('vae_checkpoint_' + str(epochs) + '.png')
MODEL_FILE = get_model_dir('vae.pth')
TRAIN_VALID_SPLIT = 0.2 # Percent of training data to be used for validation
LATENT_DIMS = 128 # Latent space dimensions
BATCH_SIZE = 256
NUM_EPOCHS = 1000
CHECKPOINT_EVERY = 1
MANUAL_SEED = 0

### Training function
def train_epoch(vae, device, dataloader, optimizer, losser):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, _ in dataloader:
        # Move tensor to the proper device
        x = x.to(device)
        # Switch batch and num channels (for some reason they are switched)
        x = x.view(-1, 1, TOTAL_TICKS, NUM_NOTES)
        x_hat = vae(x)
        # Evaluate loss
        # loss = ((x - x_hat)**2).sum() + vae.encoder.kl
        loss = losser(x_hat, x) + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(vae.parameters(),5)
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss+=loss.item()

    return train_loss / len(dataloader.dataset)

### Testing function
def test_epoch(vae, device, dataloader, losser):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x, _ in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Switch batch and num channels (for some reason they are switched)
            x = x.view(-1, 1, TOTAL_TICKS, NUM_NOTES)
            # Encode data
            encoded_data = vae.encoder(x)
            # Decode data
            x_hat = vae(x)
            # loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            loss = losser(x_hat, x) + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)

# def plot_ae_outputs(encoder,decoder,n=10):
#     plt.figure(figsize=(16,4.5))
#     targets = test_dataset.targets.numpy()
#     t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
#     for i in range(n):
#       ax = plt.subplot(2,n,i+1)
#       img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
#       encoder.eval()
#       decoder.eval()
#       with torch.no_grad():
#          rec_img  = decoder(encoder(img))
#       plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
#       ax.get_xaxis().set_visible(False)
#       ax.get_yaxis().set_visible(False)  
#       if i == n//2:
#         ax.set_title('Original images')
#       ax = plt.subplot(2, n, i + 1 + n)
#       plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
#       ax.get_xaxis().set_visible(False)
#       ax.get_yaxis().set_visible(False)  
#       if i == n//2:
#          ax.set_title('Reconstructed images')
#     plt.show()  

if __name__ == '__main__':

    inp = input('Use manual seed? (Y/N, default N)')
    if inp == 'Y' or inp == 'y':
        torch.manual_seed(MANUAL_SEED)

    inp = input('Regenerate df? (Y/N, default N)')
    regenerate_df = False
    minimal_df = False
    if inp == 'Y' or inp == 'y':
        regenerate_df = True
        inp = input('Minimal df? (Y/N, default N)')
        if inp == 'Y' or inp == 'y':
            minimal_df = True
        
    inp = input('Keep all data in memory? (Y/N, default N)')
    cache_all = False
    if inp == 'Y' or inp == 'y':
        cache_all = True
        modulo = int(input('Modulo for data (1 to use all):'))

    if regenerate_df:
        print('Regenerating df...')
        if minimal_df:
            save_data(DATA_MINIMAL_DIR, minimal=True)
        else:
            save_data(DATA_DIR)
        print('Saved df!')

    # Dataset
    def getitem(file):
        row = pd.read_pickle(file)
        x = pd_row_to_torch(row)
        return torch.tensor(x)
    if cache_all: 
        dataset = DfDataset(DATA_DIR, modulo=modulo)
    else: 
        dataset = DatasetFolder(DATA_DIR_DATASET_FOLDER, getitem, extensions=('pkl',))

    m = len(dataset)
    train_data, val_data = random_split(dataset, [m - int(m*TRAIN_VALID_SPLIT), int(m*TRAIN_VALID_SPLIT)])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # Parameterization
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    vae = VariationalAutoencoder(device, latent_dims=LATENT_DIMS)
    lr = 1e-3
    optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

    losser = BCELoss(reduction='sum')

    # Model needs to use cuda if available since data does
    if torch.cuda.is_available():
        vae.cuda()
    # Train
    train_losses = []
    val_losses = []
    print('training start')
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(vae,device,train_loader,optim, losser)
        val_loss = test_epoch(vae,device,valid_loader, losser)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, NUM_EPOCHS,train_loss,val_loss))
        # plot_ae_outputs(vae.encoder,vae.decoder,n=10)
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            torch.save(vae.state_dict(), get_model_checkpoint_file(epoch + 1))
            plt.plot(np.arange(epoch + 1), train_losses, label='Train loss')
            plt.plot(np.arange(epoch + 1), val_losses, label='Val loss')
            plt.legend()
            plt.savefig(get_loss_checkpoint_file(epoch + 1))
            plt.clf()
            print('Saved checkpoint', (epoch + 1))

    # Save model
    torch.save(vae.state_dict(), MODEL_FILE)

    # Plot losses
    plt.plot(np.arange(NUM_EPOCHS), train_losses, label='Train loss')
    plt.plot(np.arange(NUM_EPOCHS), val_losses, label='Val loss')
    plt.legend()
    plt.show()

