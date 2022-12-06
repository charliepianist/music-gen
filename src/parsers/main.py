import parsers.giant_midi as giant_midi
from parsers.midi.midi_features import MidiFeatures
import os

def _get_all_paths(minimal):
    if minimal:
        return giant_midi.get_paths(minimal)
    return giant_midi.get_paths()

def save_data(data_dir, minimal=False, start_from=0):
    """ Load all data (in feature space). This will generate splitted midis if necessary and combine
    all datasets. To use a (very) minimal subset of the datasets, for quick testing, pass minimal=True. """
    df = _get_all_paths(minimal)

    # load each midi
    for i in range(start_from, len(df)):
        row = df.iloc[i]
        
        if i % 100 == 0:
            print('saving file', i, 'of', len(df), 'at', row['path'])
        try:
            filepath = os.path.join(data_dir, 'row-' + str(i) + '.pkl')

            features = MidiFeatures(row['path'], row['midi_parser'])
            merged = features.merge_pd_row(row)
            merged.to_pickle(filepath)

            del merged
            del features
        except Exception as e:
            raise Exception('Could not parse file ' + str(i) + ' (' + row['path'] + ')')