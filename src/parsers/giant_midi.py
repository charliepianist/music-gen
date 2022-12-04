import pandas as pd
import numpy as np
from parsers.common import get_midi_dir, get_raw_dir, get_common_composers
from parsers.common import constants
from parsers.common import generate_splitted
from parsers.common import drop_other_cols, drop_important_nas, drop_raw_paths_that_dont_exist
import os

METADATA_FILE = 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv'
DIR_NAME = 'GiantMIDI'
MIN_COMPOSER_ROWS = 10 # Only count composers with this many pieces
MIDI_PARSER = 'default'

def get_raw_df():
    """
    Gets dataframe that is partially processed. Only contains raw_paths for files, not the splitted paths or split_num.
    """
    # Dataframe wrangling
    df = pd.read_csv(get_raw_dir('GiantMIDI', METADATA_FILE), delimiter='\t')
    df = df.rename(columns={
        'surname': 'last_name',
        'firstname': 'first_name',
        'birth': 'birth_year',
        'music': 'title',
        'death': 'death_year',
        'audio_name': 'raw_path',
        'nationality': 'nationality',
    })
    print('GiantMIDI initial num pieces:', len(df))

    # Not all paths are in the csv or exist
    df = drop_important_nas(df)
    df['raw_path'] = df['raw_path'].map(lambda filename : get_raw_dir(DIR_NAME, 'midi', filename + '.mid'))
    df = drop_raw_paths_that_dont_exist(df, get_raw_dir(DIR_NAME, 'midi'))

    df['midi_parser'] = MIDI_PARSER
    df['duration'] = constants['SAMPLE_DURATION']
    # path and split_num will be created after splitting midis
    df = drop_other_cols(df)

    # Filter to common composers
    df = df[df['last_name'].isin(get_common_composers(df, MIN_COMPOSER_ROWS))]
    print('GiantMIDI final num pieces:', len(df))

    return df

def get_paths(minimal=False):
    """
    Read metadata, then generate splitted MIDIs for the appropriate files, then return df with those splits
    """
    # Read in csv
    df = get_raw_df()
    
    # For testing
    if minimal:
        df = df[:5]

    # Generate splits and merge df with splits
    df = generate_splitted(df, get_midi_dir(DIR_NAME))
    return df