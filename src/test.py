import os
from parsers.midi.midi_features import MidiFeatures, save_features_to_midi, unmerge_pd_row, save_torch_to_midi, pd_row_to_torch
import pandas as pd

if __name__ == '__main__':
    dir = 'D:\\Projects\\Music Gen\\data\\GiantMIDI\\processed'
    files = os.listdir(dir)
    idx = 12
    
    mid = MidiFeatures(os.path.join(dir, files[idx]), 'giant_midi')
    row = pd.Series({'path': os.path.join(dir, files[idx]), 'test': 2})
    # print(mid)
    print(row['path'])
    merged = mid.merge_pd_row(row)
    print('======== merged:')
    print(merged)
    print('======== unmerged:')
    unmerged = unmerge_pd_row(merged)
    print(unmerged)

    # save_features_to_midi(unmerged, os.path.join(dir, 'test.mid'))
    save_torch_to_midi(pd_row_to_torch(merged), os.path.join(dir, 'test.mid'))