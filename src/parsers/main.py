import parsers.giant_midi as giant_midi

def _get_all_paths(minimal):
    if minimal:
        return giant_midi.get_paths(minimal)
    return giant_midi.get_paths()

def get_data(minimal=False):
    """ Load all data (in feature space). This will generate splitted midis if necessary and combine
    all datasets. To use a (very) minimal subset of the datasets, for quick testing, pass minimal=True. """
    all_paths = _get_all_paths(minimal)
    return all_paths