import os
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

class MidiFeatures:
    """
        Data is represented as n x 88 feature space, where
        n = SAMPLE_DURATION * TICKS_PER_SECOND
        SAMPLE_DURATION is 30 (seconds) and TICKS_PER_SECOND is 50

        Feature values are 0 if note is not being played and velocity otherwise
        Worth considering
        - Different TICKS_PER_SECOND
        - Doing stuff by beats rather than time
        - 0/1 instead of velocity and 0
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
        feature_dict = { 'feature_' + str(i * NUM_NOTES + j) : self.features[i][j] for i in range(TOTAL_TICKS) for j in range(NUM_NOTES)}
        feature_series = pd.Series(feature_dict)
        return pd.concat([row, feature_series])

    def unmerge_pd_row(self, row : pd.Series):
        """
            Undo merge from merge_pd_row
        """
        feature_series = row.filter(like='feature_')
        return np.reshape(feature_series.to_numpy(), (TOTAL_TICKS, NUM_NOTES))

    def save_midi(self, path : str):
        """
            Inefficiently save MidiFeatures as midi to path. This is intended for debugging and is NOT efficient.
        """
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
            new_velocities = self.features[tick]
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