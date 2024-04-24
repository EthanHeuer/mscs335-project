import torch
from midi_model import MidiModel

mh = MidiModel()
mh.init_data_dir()
# mh.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mh.device = "cpu"

# Training Parameters

mh.epochs = 100
mh.batch_size = 16
mh.learning_rate = 0.0005

mh.sample_range_from = 0
mh.sample_range_to = 1

# LSTM Model Parameters

mh.input_size = 512
mh.hidden_size = 256
mh.output_size = 128

# LSTM Training Loss Parameters

mh.pitch_weight = 1.0
mh.step_weight = 1.0
mh.duration_weight = 10.0

####################################################################################################################

# Create Model

mh.midi.load_files("../data/samples/input.mid")
mh.load_train_notes("base", mh.sample_range_from, mh.sample_range_to)
mh.create_model()
mh.train_model()

####################################################################################################################

# Generate Notes

input = mh.midi.files[mh.sample_range_from]
input_notes = mh.midi.get_notes(input)
num_predictions = 100
seq_length = 250
temperature = 1

generated_notes, out_pm, outfile = mh.generate_notes(input_notes, num_predictions, seq_length, temperature)
