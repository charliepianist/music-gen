import os
import numpy as np
import pandas as pd
from parsers.splitter import split_midi
import itertools

constants = {
    'DATA_DIR': 'D:\\Projects\\Music Gen\\data',
    'MIDI_DIR': 'processed',
    'RAW_DIR': 'raw',
    'SAMPLE_DURATION': 30,
}

def _get_dir(dir_name : str, *additional_dirs):
    return os.path.join(constants['DATA_DIR'], dir_name, *additional_dirs)

def get_midi_dir(dir_name : str, *additional_dirs):
    return _get_dir(dir_name, constants['MIDI_DIR'], *additional_dirs)

def get_raw_dir(dir_name : str, *additional_dirs):
    return _get_dir(dir_name, constants['RAW_DIR'], *additional_dirs)

def get_common_composers(df : pd.DataFrame, min_count):
    """ Get the composers who show up at least min_count in df (which must have a last_name column) """
    last_names = df['last_name']
    counts = last_names.value_counts()
    return list(counts[counts >= min_count].index)

def drop_other_cols(df : pd.DataFrame, *additional_cols : list[str]) -> pd.DataFrame:
    """ Keep only the relevant columns (i.e. the ones in the README, plus raw_path) """
    relevant_cols = {
        'last_name', 'first_name', 'nationality', 'birth_year', 'death_year', 'title', 
        'path', 'split_num', 'midi_parser', 'duration', 'raw_path'
        }
    relevant_cols = relevant_cols.union(additional_cols)
    relevant_cols = relevant_cols.intersection(df.columns)
    return df[list(relevant_cols)]
    
def drop_important_nas(df : pd.DataFrame) -> pd.DataFrame:
    """ Drop rows that have NaN/Null for important columns """
    important_cols = ['last_name', 'raw_path']
    return df.dropna(subset=important_cols)

def drop_raw_paths_that_dont_exist(df : pd.DataFrame, dir : str) -> pd.DataFrame:
    """ Drop rows for which their raw_path does not exist """
    existing_files = [os.path.join(dir, path) for path in os.listdir(dir)]
    return df[df['raw_path'].isin(existing_files)]

def generate_splitted(df: pd.DataFrame, target_folder : str) -> pd.DataFrame:
    """ Generate splitted midis for all files in a raw df and return original df along with
        'path' column that contains splitted paths """
    tails = df['raw_path'].map(lambda raw_path : os.path.split(raw_path)[1])
    df['_base'] = tails.map(lambda tail : tail[:tail.index('.')])
    for i in range(len(df)):
        row = df.iloc[i]
        basename = row['_base']

        # Don't re-split if already donw
        if not os.path.isfile(os.path.join(target_folder, basename + '-split-1.mid')):
            split_midi(row['raw_path'], target_folder, row['duration'])
    
    splitted_and_base = pd.DataFrame({'path': os.listdir(target_folder)})
    # Split found directories into base and the split num
    components = splitted_and_base['path'].str.rsplit('-split', n=1, expand=True)
    splitted_and_base['_base'] = components[0]
    splitted_and_base['split_num'] = components[1].str.slice(start=1, stop=-4)

    # Filter to relevant files
    splitted_and_base = splitted_and_base[splitted_and_base['_base'].isin(df['_base'])]

    # Join path back with target_folder
    splitted_and_base['path'] = splitted_and_base['path'].map(lambda path : os.path.join(target_folder, path))

    # Merge and return
    df_merged = splitted_and_base.merge(df, how='left', on='_base')
    return df_merged.drop(columns=['_base'])