import torch
import midi_handler as mh
import numpy as np
import torch.nn as nn
from network import Data, LSTMModel

# LOAD FILES

handler = mh.MidiHandler()
handler.load_files("../data/maestro/**/*.mid*")
handler.load_files("../data/Cymatics/*.mid*")

print(f"Number of Samples: {len(handler.files)}")


# PLOT MIDI PIANO ROLL

# handler.plot_piano_roll(handler.files[0])


# GET NOTES FROM SAMPLES

all_notes = handler.get_notes_from_samples(0, 5)
key_order = ["pitch", "step", "duration"]
all_notes.to_csv("../data/notes-output.csv", index=False)
train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
np.save("../data/notes-output.npy", train_notes)

print(train_notes.shape)

input_size = 3
hidden_size = 128
output_size = 128

model = LSTMModel(input_size, hidden_size, output_size)
model.cuda()

criterion = {
    "pitch": nn.CrossEntropyLoss(),
    "step": nn.MSELoss(),
    "duration": nn.MSELoss(),
}
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
