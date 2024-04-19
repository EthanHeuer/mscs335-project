import torch
from midi_model import MidiModel


######################################################################################
# VARIABLES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mh = MidiModel(device)
print(f"Running on: {device}")


######################################################################################
# LOAD FILES

mh.midi.load_files("../data/maestro/**/*.mid*")
# mh.midi.load_files("../data/Cymatics/*.mid*")
print(f"Number of Samples: {len(mh.midi.files)}")


######################################################################################
# GET NOTES FROM SAMPLES

train_notes = mh.load_train_notes("../data")
print("Number of notes parsed:", len(train_notes))


######################################################################################
# SETUP MODEL

epochs = 50
batch_size = 16
learning_rate = 0.0005

data, loader, model, criterion, optimizer = mh.create_model(epochs, batch_size, learning_rate)


######################################################################################
# TRAIN MODEL

mh.train_model()


######################################################################################
# PREDICT NOTES

raw_notes = mh.midi.get_notes(mh.midi.files[0])
num_predictions = 120
seq_length = 25
temperature = 5.0

generated_notes, out_pm = mh.generate_notes(raw_notes, num_predictions, seq_length, temperature)

# mh.midi.display_audio(out_pm)
# handler.plot_piano_roll(generated_notes)
