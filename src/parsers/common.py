import os

constants = {
    'DATA_DIR': 'D:\\Projects\\Music Gen\\data',
    'MIDI_DIR': 'processed',
    'RAW_DIR': 'raw'
}

def _get_dir(dir_name : str, *additional_dirs):
    return os.path.join(constants['DATA_DIR'], dir_name, *additional_dirs)

def get_midi_dir(dir_name : str):
    return _get_dir(dir_name, constants['MIDI_DIR'])

def get_raw_dir(dir_name : str):
    return _get_dir(dir_name, constants['RAW_DIR'])