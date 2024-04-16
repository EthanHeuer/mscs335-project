import time
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from midi_handler import MidiHandler
from model import LSTMModel, DataHandler


######################################################################################
# VARIABLES

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0005

######################################################################################
# SETUP DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {device}")


######################################################################################
# LOAD FILES

handler = MidiHandler()
handler.load_files("../data/maestro/**/*.mid*")
handler.load_files("../data/Cymatics/*.mid*")

print(f"Number of Samples: {len(handler.files)}")


######################################################################################
# GET NOTES FROM SAMPLES

all_notes = handler.get_notes_from_samples(0, 1282)
key_order = ["pitch", "step", "duration"]
train_notes = torch.tensor(all_notes[key_order].values, dtype=torch.float32).to(device)
all_notes.to_csv("../data/notes-output.csv", index=False)

print("Number of notes parsed:", len(all_notes))


######################################################################################
# SETUP MODEL

data = DataHandler(train_notes)
loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

model = LSTMModel(3, 128, 128)
model.to(device)

criterion = {
    "pitch": nn.CrossEntropyLoss(),
    "step": nn.MSELoss(),
    "duration": nn.MSELoss(),
}
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


######################################################################################
# TRAIN MODEL

start = time.time()

for epoch in range(EPOCHS):
    start_epoch = time.time()
    running_loss = 0.0

    for i, item in enumerate(loader, 0):
        input = item.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = (
            criterion["pitch"](output["pitch"], input[:, 0].long())
            + criterion["step"](output["step"], input[:, 1].view(-1, 1))
            + criterion["duration"](output["duration"], input[:, 2].view(-1, 1))
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(input)

    end_epoch = time.time()
    print(
        f"Epoch {epoch + 1}/{EPOCHS}: {running_loss / len(data)} ({end_epoch - start_epoch} seconds)"
    )

stop = time.time()

print(f"Training time: {stop - start} seconds")

torch.save(model.state_dict(), "../models/model.pt")


######################################################################################
# PREDICT NOTES


def predict_next_note(notes, model, temperature=1.0):
    inputs = torch.tensor(notes, dtype=torch.float32).unsqueeze(0).to(device)
    outputs = model(inputs)
    pitch_logits = outputs["pitch"]
    step = outputs["step"]
    duration = outputs["duration"]

    pitch_probs = torch.softmax(pitch_logits / temperature, dim=2)

    pitch = torch.multinomial(pitch_probs[0, -1], 1).item()
    step = step[0, -1].item()
    duration = duration[0, -1].item()

    return pitch, step, duration


raw_notes = handler.get_midi_notes(handler.files[0])

temperature = 2.0
num_predictions = 120
seq_length = 30

sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
input_notes = sample_notes[:seq_length]

generated_notes = []
prev_start = 0
for _ in range(num_predictions):
    pitch, step, duration = predict_next_note(input_notes, model, temperature)
    start = prev_start + step
    end = start + duration
    input_note = (pitch, step, duration)
    generated_notes.append((*input_note, start, end))
    input_notes = np.delete(input_notes, 0, axis=0)
    input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    prev_start = start

generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, "start", "end"))

out_file = "output.mid"
out_pm = handler.notes_to_midi(
    generated_notes, out_file=out_file, instrument_name="Acoustic Grand Piano"
)
handler.display_audio(out_pm)
# handler.plot_piano_roll(generated_notes)
