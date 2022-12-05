import os
from parsers.midi.midi_features import MidiFeatures
import pandas as pd

if __name__ == '__main__':
    dir = 'D:\\Projects\\Music Gen\\data\\GiantMIDI\\processed'
    files = os.listdir(dir)
    idx = 12
    
    mid = MidiFeatures(os.path.join(dir, files[idx]), 'giant_midi')
    row = pd.Series({'path': os.path.join(dir, files[idx]), 'test': 2})
    # print(mid)
    merged = mid.merge_pd_row(row)
    print('======== merged:')
    print(merged)
    print('======== unmerged:')
    print(mid.unmerge_pd_row(merged))

    mid.save_midi(os.path.join(dir, 'test.mid'))