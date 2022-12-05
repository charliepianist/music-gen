import parsers.giant_midi as giant_midi
from parsers.midi.midi_features import MidiFeatures

def _get_all_paths(minimal):
    if minimal:
        return giant_midi.get_paths(minimal)
    return giant_midi.get_paths()

def get_data(minimal=False):
    """ Load all data (in feature space). This will generate splitted midis if necessary and combine
    all datasets. To use a (very) minimal subset of the datasets, for quick testing, pass minimal=True. """
    df = _get_all_paths(minimal)
    
    # df row to features
    def row_to_features(row):
        features = MidiFeatures(row['path'], row['midi_parser'])
        return features.merge_pd_row(row)

    # load each midi
    df = df.apply(row_to_features, axis='columns', result_type='expand')

    return df