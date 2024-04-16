import collections
import datetime
import glob
import numpy as np
import pandas as pd
import pathlib
import pretty_midi
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from IPython import display
from matplotlib import pyplot as plt
from typing import Optional

seed = 42
batch_size = 64
learning_rate = 0.001
torch.manual_seed(seed)
np.random.seed(seed)

_SAMPLING_RATE = 16000

data_dir = pathlib.Path('data/maestro-v2.0.0')
if not data_dir.exists():
    # Use torch's download utility
    torch.hub.download_url_to_file(
        'https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
        'maestro-v2.0.0-midi.zip',
    )
    # Unzip the file (assuming you have unzip installed)
    import zipfile
    with zipfile.ZipFile('maestro-v2.0.0-midi.zip', 'r') as zip_ref:
        zip_ref.extractall('data')

filenames = glob.glob(str(data_dir / '**/*.mid*'))
print('Number of files:', len(filenames))

sample_file = filenames[1]
print(sample_file)

pm = pretty_midi.PrettyMIDI(sample_file)

def display_audio(pm: pretty_midi.PrettyMIDI, seconds=30):
    waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
    waveform_short = waveform[:seconds * _SAMPLING_RATE]
    return display.Audio(waveform_short, rate=_SAMPLING_RATE)

display_audio(pm)

print('Number of instruments:', len(pm.instruments))
instrument = pm.instruments[0]
instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
print('Instrument name:', instrument_name)

for i, note in enumerate(instrument.notes[:10]):
    note_name = pretty_midi.note_number_to_name(note.pitch)
    duration = note.end - note.start
    print(f'{i}: pitch={note.pitch}, note_name={note_name}, duration={duration:.4f}')

def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

raw_notes = midi_to_notes(sample_file)
raw_notes.head()

get_note_names = np.vectorize(pretty_midi.note_number_to_name)
sample_note_names = get_note_names(raw_notes['pitch'])
sample_note_names[:10]

def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
    if count:
        title = f'First {count} notes'
    else:
        title = f'Whole track'
        count = len(notes['pitch'])
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(
        plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    _ = plt.title(title)

plot_piano_roll(raw_notes, count=100)

plot_piano_roll(raw_notes)

def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    sns.histplot(notes, x="pitch", bins=20)

    plt.subplot(1, 3, 2)
    max_step = np.percentile(notes['step'], 100 - drop_percentile)
    sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))

    plt.subplot(1, 3, 3)
    max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
    sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))

plot_distributions(raw_notes)

def notes_to_midi(
    notes: pd.DataFrame,
    out_file: str,
    instrument_name: str,
    velocity: int = 100,
) -> pretty_midi.PrettyMIDI:

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm

example_file = 'example.midi'
example_pm = notes_to_midi(
    raw_notes, out_file=example_file, instrument_name=instrument_name)

display_audio(example_pm)

num_files = 5
all_notes = []
for f in filenames[:num_files]:
    notes = midi_to_notes(f)
    all_notes.append(notes)

all_notes = pd.concat(all_notes)

n_notes = len(all_notes)
print('Number of notes parsed:', n_notes)

key_order = ['pitch', 'step', 'duration']
train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

notes_ds = torch.utils.data.TensorDataset(torch.tensor(train_notes, dtype=torch.float32))
notes_dl = torch.utils.data.DataLoader(notes_ds, batch_size=batch_size, shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.pitch_fc = nn.Linear(hidden_size, output_size)
        self.step_fc = nn.Linear(hidden_size, 1)
        self.duration_fc = nn.Linear(hidden_size, 1)

    def forward(self, input):
        output, _ = self.lstm(input)
        pitch_out = self.pitch_fc(output)
        step_out = self.step_fc(output)
        duration_out = self.duration_fc(output)
        return {
            'pitch': pitch_out,
            'step': step_out,
            'duration': duration_out,
        }

input_size = 3
hidden_size = 128
output_size = 128

model = LSTMModel(input_size, hidden_size, output_size)

criterion = {
    'pitch': nn.CrossEntropyLoss(),
    'step': nn.MSELoss(),
    'duration': nn.MSELoss(),
}
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train(model, dataloader, criterion, optimizer, num_epochs=50):
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch[0].unsqueeze(0)  # Add batch dimension
            targets = {
                'pitch': batch[0][:, 0],  # Pitch is the first element
                'step': batch[0][:, 1].unsqueeze(1),  # Step is the second element
                'duration': batch[0][:, 2].unsqueeze(1),  # Duration is the third element
            }
            outputs = model(inputs)
            loss = sum(criterion[key](outputs[key], targets[key]) for key in criterion)
            loss.backward()
            optimizer.step()

train(model, notes_dl, criterion, optimizer)

def predict_next_note(
    notes: np.ndarray,
    model: LSTMModel,
    temperature: float = 1.0) -> tuple[int, float, float]:

    inputs = torch.tensor(notes, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    outputs = model(inputs)
    pitch_logits = outputs['pitch']
    step = outputs['step']
    duration = outputs['duration']

    pitch_logits /= temperature
    pitch = torch.multinomial(torch.softmax(pitch_logits.squeeze(), dim=0), num_samples=1).item()
    step = step.squeeze().item()
    duration = duration.squeeze().item()

    step = max(0, step)
    duration = max(0, duration)

    return pitch, step, duration

temperature = 2.0
num_predictions = 120

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

generated_notes = pd.DataFrame(
    generated_notes, columns=(*key_order, 'start', 'end'))

out_file = 'output.mid'
out_pm = notes_to_midi(
    generated_notes, out_file=out_file, instrument_name=instrument_name)
display_audio(out_pm)

plot_piano_roll(generated_notes)

plot_distributions(generated_notes)
