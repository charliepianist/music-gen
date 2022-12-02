from lib.midi_orchestra_master.common import get_files, check_target_folder, is_invalid_file, make_file_path
import pretty_midi as midi
import math

def _find_elements_in_range(elements, start_time, end_time):
    """Filters elements which are within a time range."""

    filtered_elements = []

    for item in elements:
        if hasattr(item, 'start') and hasattr(item, 'end'):
            start = item.start
            end = item.end
        elif hasattr(item, 'time'):
            start = item.time
            end = item.time

        if not (end <= start_time or start >= end_time):
            if hasattr(item, 'start') and hasattr(item, 'end'):
                item.start = item.start - start_time
                item.end = item.end - start_time
            elif hasattr(item, 'time'):
                item.time = item.time - start_time

            filtered_elements.append(item)

    return filtered_elements

def _split_score(score, split_every_sec):
    """Break the MIDI file into smaller parts."""

    end_time = score.get_end_time()

    # Get instruments
    instruments = score.instruments

    # Get all time signature changes
    time_signature_changes = score.time_signature_changes

    # Get all key changes
    key_changes = score.key_signature_changes

    last_time_signature_change = None
    if len(time_signature_changes) > 0:
        last_time_signature_change = time_signature_changes[0]

    last_key_change = None
    if len(key_changes) > 0:
        last_key_change = key_changes[0]

    splits = []

    # Split score into smaller time spans
    for split_index, split_start_time in enumerate(range(0,
                                                         math.ceil(end_time),
                                                         split_every_sec)):
        split_end_time = min(split_start_time + split_every_sec, end_time)

        split_instruments = []
        split_notes_counter = 0

        print('Generate split #{} from {} sec - {} sec.'.format(
            split_index + 1, split_start_time, split_end_time))

        for instrument in instruments:
            # Find notes for this instrument in this range
            split_notes = _find_elements_in_range(instrument.notes,
                                                 split_start_time,
                                                 split_end_time)

            split_notes_counter += len(split_notes)

            # Create new instrument
            split_instrument = midi.Instrument(program=instrument.program,
                                               name=instrument.name)

            split_instrument.notes = split_notes
            split_instruments.append(split_instrument)

        # Find key and time signature changes
        split_time_changes = _find_elements_in_range(time_signature_changes,
                                                    split_start_time,
                                                    split_end_time)

        if len(split_time_changes) > 0:
            last_time_signature_change = split_time_changes[-1]
        elif last_time_signature_change:
            split_time_changes = [last_time_signature_change]

        split_key_signature_changes = _find_elements_in_range(key_changes,
                                                             split_start_time,
                                                             split_end_time)

        if len(split_key_signature_changes) > 0:
            last_key_change = split_key_signature_changes[-1]
        elif last_key_change:
            split_key_signature_changes = [last_key_change]

        print('Found {} notes, '
              'added {} key changes and '
              '{} time signature changes.'.format(
                  split_notes_counter,
                  len(split_key_signature_changes),
                  len(split_time_changes)))

        splits.append({'instruments': split_instruments,
                       'time_signature_changes': split_time_changes,
                       'key_signature_changes': split_key_signature_changes})

    return splits


def _generate_files(file_path, target_folder, splits):
    """Saves multiple splitted MIDI files in a folder."""

    for split_index, split in enumerate(splits):
        split_score = midi.PrettyMIDI()
        split_score.time_signature_changes = split['time_signature_changes']
        split_score.key_signature_changes = split['key_signature_changes']
        split_score.instruments = split['instruments']

        # Save MIDI file
        split_file_path = make_file_path(
            file_path, target_folder,
            suffix='split-{}'.format(split_index + 1))
        split_score.write(split_file_path)

        print('Saved MIDI file at "{}".'.format(split_file_path))

def split_midi(file : str, target_folder : str, duration : int):
    check_target_folder(target_folder)

    if is_invalid_file(file):
        return

    # Read MIDi file and clean up
    score = midi.PrettyMIDI(file)
    score.remove_invalid_notes()

    # Split MIDI file!
    splits = _split_score(score, duration)

    # Generate MIDI files from splits
    _generate_files(file, target_folder, splits)