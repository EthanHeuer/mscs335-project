import midi_handler as mh
import numpy as np

# LOAD FILES

handler = mh.MidiHandler()
handler.load_files("../data/maestro/**/*.mid*")
handler.load_files("../data/Cymatics/*.mid*")

print(f"Number of Samples: {len(handler.files)}")

# PLOT MIDI PIANO ROLL

# handler.plot_piano_roll(handler.files[0])

# GET NOTES FROM SAMPLES

all_notes = handler.get_notes_from_samples(1282)
key_order = ["pitch", "step", "duration"]
all_notes.to_csv("../data/notes-output.csv", index=False)
train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
np.save("../data/notes-output.npy", train_notes)