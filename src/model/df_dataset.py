from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from parsers.midi.midi_features import pd_row_to_torch
import os

class DfDataset(Dataset):
    """
        Cache ENTIRE dataset in memory. This is much faster for training but is memory constrained
    """
    def __init__(self, data_dir, modulo=1):
        self.data_dir = data_dir

        files = os.listdir(data_dir)
        num_rows = len(files)
        all_rows = []
        for row_num in range(num_rows):
            if row_num % modulo == 0:
                row = pd.read_pickle(os.path.join(self.data_dir, 'row-' + str(row_num) + '.pkl'))
                x = pd_row_to_torch(row)
                all_rows.append(x)

        self.x_train = torch.tensor(np.array(all_rows))

    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self,idx):
        return self.x_train[idx], 0