import torch
from mido import MidiFile, MidiTrack, MetaMessage, Message, tick2second, second2tick, merge_tracks
import pandas as pd
import numpy as np
from collections import defaultdict
from parsers.common import constants

MIN_NOTE = 21 # Lowest note on a piano has this note #
NUM_NOTES = 88

SAMPLE_DURATION = constants['SAMPLE_DURATION']
TICKS_PER_SECOND = 25
TOTAL_TICKS = SAMPLE_DURATION * TICKS_PER_SECOND

FLATTENED_FEATURES = True

MAX_VELOCITY = 128

# What value should be allowed to _start_ a note?
RECONSTRUCT_THRESHOLD = 0.1
# What value must be maintained to consider it as still a note?
RECONSTRUCT_CONTIGUOUS_THRESHOLD = 0.05
# Linearly interpolate velocity from average value of features
RECONSTRUCT_MIN_VELOCITY = 50
RECONSTRUCT_MAX_VELOCITY = 100

class MidiFeatures:
    """
        Data is represented as n x 88 feature space, where
        n = SAMPLE_DURATION * TICKS_PER_SECOND
        SAMPLE_DURATION is 30 (seconds) and TICKS_PER_SECOND is 50

        Feature values are 0 if note is not being played and velocity otherwise
        Worth considering
        - Different TICKS_PER_SECOND
        - Doing stuff by beats rather than time

        Note that these are intended to be transformed via pd_row_to_torch, which makes velocities all 0 or 1
        Consider making times 0.8, 0.6, 0.4, 0.2 around start and end of each note.
    """
    def __init__(self, path : str, parser : str):
        mid = MidiFile(path, clip=True)
        # This should never actually be the values for features, but it is the right shape
        features = np.zeros(shape=(TOTAL_TICKS, NUM_NOTES)) - 1 

        # Choose appropriate parser
        match parser:
            case 'giant_midi':
                features = giant_midi_features_of_midi(mid)
            case other:
                raise Exception('Missing parser ' + other)

        # Set features
        self.features = features.astype('int')

    def __str__(self):
        """ For debugging """
        return str({
            'features': self.features
        })

    def merge_pd_row(self, row : pd.Series): 
        """
            Merge row with features
        """
        if FLATTENED_FEATURES:
            feature_dict = { 'feature_' + str(i * NUM_NOTES + j) : self.features[i][j] for i in range(TOTAL_TICKS) for j in range(NUM_NOTES)}
        else:
            feature_dict = { 'feature_' + str(i) : self.features[i] for i in range(TOTAL_TICKS) }
        feature_series = pd.Series(feature_dict)
        return pd.concat([row, feature_series])

    def save_midi(self, path : str):
        """
            Inefficiently save MidiFeatures as midi to path. This is intended for debugging and is NOT efficient.
        """
        save_features_to_midi(self.features, path)
        

def save_features_to_midi(features, path: str):
    tempo = 500000 # microseconds per beat
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    mid.ticks_per_beat = 220 # Idk this is just what was used in GiantMIDI
    track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    ref_velocities = np.zeros((NUM_NOTES,))
    last_time = 0 # In MIDI time

    # just changes in velocities
    for tick in range(TOTAL_TICKS):
        new_velocities = np.array(features[tick]).astype('int')
        cur_time = round(second2tick(tick / TICKS_PER_SECOND, mid.ticks_per_beat, tempo))
        # Iterate through all notes
        for note_num in range(NUM_NOTES):
            if ref_velocities[note_num] != new_velocities[note_num]:
                # Time is only nonzero for the first note
                time = cur_time - last_time
                last_time = cur_time
                track.append(Message('note_on', note=note_num + MIN_NOTE, velocity=new_velocities[note_num], time=time))
        ref_velocities = new_velocities
    
    # Mark end of track
    track.append(MetaMessage('end_of_track', time=1))

    # Save file
    mid.save(path)

def save_torch_to_midi(features, path:str, reconstruct_thresh=RECONSTRUCT_THRESHOLD, reconstruct_contig_thresh=RECONSTRUCT_CONTIGUOUS_THRESHOLD, min_vel=RECONSTRUCT_MIN_VELOCITY, max_vel=RECONSTRUCT_MAX_VELOCITY):
    features = torch.sigmoid(features)
    features = np.array(features)
    note_on = np.full((NUM_NOTES,), False)
    ends = np.full((NUM_NOTES,), -1) # Where current string ends
    velocities = np.full((NUM_NOTES,), -1) # what velocity to fill with

    # Iterate 
    for tick in range(TOTAL_TICKS):
        for note in range(NUM_NOTES):
            if features[tick][note] >= reconstruct_thresh:
                # print('test')
                # Start of a note, compute where it ends and its average velocity
                note_on[note] = True
                idx = tick
                total_vel = 0
                while idx < TOTAL_TICKS and features[idx][note] > reconstruct_contig_thresh:
                    total_vel += features[idx][note]
                    idx += 1
                ends[note] = idx - 1
                velocities[note] = (total_vel / (idx - tick)) * (max_vel - min_vel) + min_vel
                # print(velocities[note], tick, note)

            # Write note if relevant
            if note_on[note]:
                features[tick][note] = velocities[note]
                if ends[note] == tick:
                    # Note ends
                    ends[note] = -1
                    note_on[note] = False
            else:
                features[tick][note] = 0
    return save_features_to_midi(features, path)

def unmerge_pd_row(row : pd.Series):
    """
        Undo merge from merge_pd_row
    """
    feature_series = row.filter(like='feature_').astype('float32')
    if FLATTENED_FEATURES:
        return np.reshape(feature_series.to_numpy(), (TOTAL_TICKS, NUM_NOTES))
    else:
        return np.reshape(np.stack(feature_series.to_numpy()), (TOTAL_TICKS, NUM_NOTES))

def pd_row_to_torch(row : pd.Series):
    """
        Undo merge from merge_pd_row
    """
    features = np.array(unmerge_pd_row(row))
    features[features != 0] = 1

    return features
    # return features / MAX_VELOCITY

# ================================= Helpers ==========================================
def giant_midi_features_of_midi(mid):
    """
        Notable features of GiantMIDI files:
        - Tempo is always 500000 (the default, 500000 microseconds per beat i.e. 120 bpm)
            - This is stored in track 0
        - All notes in the piece occur in track 1
        - Note off is notated as note_on velocity=0 rather than note_off
    """
    return default_features_of_midi(mid)

def default_features_of_midi(mid):
    """
        Merges tracks, and assumes that set_tempo time uses _old_ tempo
    """
    cur_time = 0 # Current time (in seconds, approx)
    note_velocities = defaultdict(lambda : 0) # 0 if not on, otherwise velocity
    last_tick = 0 # This is the most recent tick that needs to be filled
    features = np.zeros(shape=(TOTAL_TICKS, NUM_NOTES))

    tempo = 500000 # microseconds per beat

    # feature vector for one time, from note_velocities
    def velocities_arr_of_note_velocities(nv):
        return [nv[i] for i in range(NUM_NOTES)]

    # Get note velocities and times
    for message in merge_tracks(mid.tracks):
        msg = message.dict()

        # add time to cur_time (seconds)
        time = tick2second(msg['time'], mid.ticks_per_beat, tempo)
        cur_time += time

        # Record tempo
        if msg['type'] == 'set_tempo':
            tempo = msg['tempo']
            continue
        # Ignore all messages except note_on and note_off
        if msg['type'] != 'note_on' and msg['type'] != 'note_off':
            continue

        # Backfill features up until this tick - 1 (round time to tick)
        # NOTE: this clamps the MIDI to end at SAMPLE_DURATION. This should be fine,
        # as long as samples are indeed that long.
        cur_tick = min(round(cur_time * TICKS_PER_SECOND), TOTAL_TICKS)
        if cur_tick > last_tick:
            features[last_tick:cur_tick] = np.broadcast_to(
                velocities_arr_of_note_velocities(note_velocities), 
                (cur_tick - last_tick, NUM_NOTES))
        last_tick = cur_tick

        # Keep track of velocity
        if msg['type'] == 'note_on':
            note_velocities[msg['note'] - MIN_NOTE] = msg['velocity']
        else:
            note_velocities[msg['note'] - MIN_NOTE] = 0
        
    # Backfill to the end
    if TOTAL_TICKS > last_tick:
        features[last_tick:] = np.broadcast_to(
            velocities_arr_of_note_velocities(note_velocities), 
            (TOTAL_TICKS - last_tick, NUM_NOTES))
    return features