from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from parsers.midi.midi_features import TOTAL_TICKS, NUM_NOTES

class DfDataset(Dataset):
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        
        x = df[[column for column in df.columns if 'feature_' in column]].to_numpy()
        x = np.reshape(x, (len(df), TOTAL_TICKS, NUM_NOTES))

        self.x_train = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self,idx):
        # 0 is arbitrary label
        return self.x_train[idx], 0