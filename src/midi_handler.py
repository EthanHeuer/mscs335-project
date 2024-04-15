import glob
import collections
import pretty_midi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Parsing MIDI file code implemented from:
# https://www.tensorflow.org/tutorials/audio/music_generation


class MidiHandler:
    def __init__(self):
        self.files = []
        self.notes = []

    def load_files(self, path):
        files = glob.glob(path)
        self.files.extend(files)
        print(f"Found {len(files)} files in {path}")

    def get_midi_notes(self, midi_file):
        pm = pretty_midi.PrettyMIDI(midi_file)
        instrument = pm.instruments[0]
        notes = collections.defaultdict(list)

        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            notes["pitch"].append(note.pitch)
            notes["start"].append(start)
            notes["end"].append(end)
            notes["step"].append(start - prev_start)
            notes["duration"].append(end - start)
            prev_start = start

        return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

    def plot_piano_roll(self, sample):
        notes = self.get_midi_notes(sample)

        plt.figure(figsize=(20, 4))
        plot_pitch = np.stack([notes["pitch"], notes["pitch"]], axis=0)
        plot_start_stop = np.stack([notes["start"], notes["end"]], axis=0)
        plt.plot(plot_start_stop, plot_pitch, color="b", marker=".")
        plt.xlabel("Time [s]")
        plt.ylabel("Pitch")
        plt.title(sample)

    def get_notes_from_samples(self, range_from=0, range_to=None):
        all_notes = []
        file_range = self.files[range_from:range_to]
        num_files = len(file_range)
        errors = []
        for index, file in enumerate(file_range):
            try:
                notes = self.get_midi_notes(file)
                all_notes.append(notes)
                print(f"Loaded {index + 1}/{num_files}: {file}")
            except Exception as e:
                errors.append(file)
                print(f"Error loading {file}: {e}")

        if errors:
            print(f"Errors: {errors}")

        self.notes = pd.concat(all_notes)
        return self.notes
