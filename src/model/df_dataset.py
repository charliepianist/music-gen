from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from parsers.midi.midi_features import unmerge_pd_row
import os

class DfDataset(Dataset):
    """
        Cache ENTIRE dataset in memory. This is much faster for training but is memory constrained
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

        files = os.listdir(data_dir)
        num_rows = len(files)
        all_rows = []
        for row_num in range(num_rows):
            row = pd.read_pickle(os.path.join(self.data_dir, 'row-' + str(row_num) + '.pkl'))
            x = unmerge_pd_row(row)
            all_rows.append(x)

        self.x_train = torch.tensor(np.array(all_rows)).to_sparse()

    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self,idx):
        return self.x_train[idx], 0