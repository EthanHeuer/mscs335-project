import glob
import collections
import pretty_midi
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython import display


class MidiHandler:
    def __init__(self):
        r"""
        This class is used to load, manage, and handle MIDI files.
        https://www.tensorflow.org/tutorials/audio/music_generation
        """

        self.files = []
        self.notes = []
        self._SAMPLING_RATE = 16000

    ####################################################################################################################

    def load_files(self, path: str) -> list[str]:
        r"""
        Loads MIDI files from a given path.
        """

        files = glob.glob(path)
        self.files.extend(files)
        print(f"Found {len(files)} files in {path}")

    ####################################################################################################################

    def get_notes(self, file_name: str) -> pd.DataFrame:
        r"""
        Gets notes from a single MIDI file.
        """

        pm = pretty_midi.PrettyMIDI(file_name)
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

    ####################################################################################################################

    def get_notes_from_range(self, range_from: int = 0, range_to: int | None = None) -> pd.DataFrame:
        r"""
        Gets notes from a range of samples in the files list. Files can be loaded with `load_files`.
        """

        all_notes = []
        file_range = self.files[range_from:range_to]
        num_files = len(file_range)
        errors = []
        for index, file in enumerate(file_range):
            try:
                notes = self.get_notes(file)
                all_notes.append(notes)
                print(f"Loaded {index + 1}/{num_files}: {file}")
            except Exception as e:
                errors.append(file)
                print(f"Error loading {file}: {e}")

        if errors:
            print(f"Errors: {errors}")

        self.notes = pd.concat(all_notes)
        return self.notes

    ####################################################################################################################

    def notes_to_midi(self, notes, out_file: str, instrument_name: str, velocity: int = 100) -> pretty_midi.PrettyMIDI:
        r"""
        TODO
        """

        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

        prev_start = 0
        for i, note in notes.iterrows():
            start = float(prev_start + note["step"])
            end = float(start + note["duration"])
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=int(note["pitch"]),
                start=start,
                end=end,
            )
            instrument.notes.append(note)
            prev_start = start

        pm.instruments.append(instrument)
        pm.write(out_file)
        return pm

    ####################################################################################################################

    def display_midi(self, file_name: str):
        r"""
        Plots a piano roll of the notes in a sample.
        """

        notes = self.get_notes(file_name)

        plt.figure(figsize=(20, 4))
        plot_pitch = np.stack([notes["pitch"], notes["pitch"]], axis=0)
        plot_start_stop = np.stack([notes["start"], notes["end"]], axis=0)
        plt.plot(plot_start_stop, plot_pitch, color="b", marker=".")
        plt.xlabel("Time (s)")
        plt.ylabel("Pitch")
        plt.title(file_name)
        plt.show()

    ####################################################################################################################

    def display_audio(self, pm, seconds=30):
        r"""
        TODO
        """

        waveform = pm.fluidsynth(fs=self._SAMPLING_RATE)
        waveform_short = waveform[: seconds * self._SAMPLING_RATE]
        return display.Audio(waveform_short, rate=self._SAMPLING_RATE)
