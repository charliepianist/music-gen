from parsers.main import get_data
from model.vae import VariationalAutoencoder
import torch
import matplotlib.pyplot as plt
from parsers.common import get_data_dir
from model.df_dataset import DfDataset
from torch.utils.data import DataLoader, random_split
from parsers.midi.midi_features import TOTAL_TICKS, NUM_NOTES

DF_FILE = get_data_dir('data.csv')
TRAIN_VALID_SPLIT = 0.2 # Percent of training data to be used for validation
LATENT_DIMS = 50 # Latent space dimensions
BATCH_SIZE = 256

### Training function
def train_epoch(vae, device, dataloader, optimizer):
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
        loss = ((x - x_hat)**2).sum() + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss+=loss.item()

    return train_loss / len(dataloader.dataset)

### Testing function
def test_epoch(vae, device, dataloader):
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
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
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

    inp = input('Regenerate df? (Y/N, default N)')
    if inp == 'Y' or inp == 'y':
        print('Regenerating df...')
        df = get_data(minimal=True)
        df.to_csv(DF_FILE)
        print('Saved df!')

    inp = input('Use manual seed? (Y/N, default N)')
    if inp == 'Y' or inp == 'y':
        torch.manual_seed(0)

    # Dataset
    dataset = DfDataset(DF_FILE)
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

    if torch.cuda.is_available():
        vae.cuda()
    # Train
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train_epoch(vae,device,train_loader,optim)
        val_loss = test_epoch(vae,device,valid_loader)
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
        # plot_ae_outputs(vae.encoder,vae.decoder,n=10)

