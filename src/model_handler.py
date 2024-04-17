import pathlib
import time
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from model import LSTMModel, DataHandler
from midi_handler import MidiHandler
from model import DataHandler


class ModelHandler:
    """
    This class is used to handle the model and the training data.
    """

    def __init__(self, device):
        self.midi = MidiHandler()
        self.device = device
        self.train_notes = None
        self.epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.data = None
        self.loader = None
        self.model = None
        self.criterion = None
        self.optimizer = None

    def get_train_notes(self, path, range_from=0, range_to=None):
        notes_file = pathlib.Path(path)

        if not notes_file.exists():
            print("Parsing notes from samples")

            all_notes = self.midi.get_notes_from_range(range_from, range_to)
            key_order = ["pitch", "step", "duration"]
            self.train_notes = torch.tensor(
                all_notes[key_order].values, dtype=torch.float32
            ).to(self.device)
            torch.save(self.train_notes, notes_file)
        else:
            print("Loading notes from file")

            self.train_notes = torch.load(notes_file)

        return self.train_notes

    def create_model(self, epochs: int, batch_size: int, learning_rate: float):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.data = DataHandler(self.train_notes)
        self.loader = DataLoader(self.data, batch_size=batch_size, shuffle=True)

        self.model = LSTMModel(3, 128, 128)
        self.model.to(self.device)

        self.criterion = {
            "pitch": nn.CrossEntropyLoss(),
            "step": nn.MSELoss(),
            "duration": nn.MSELoss(),
        }
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        print(
            f"Training Model: epochs={epochs}, batch size={batch_size}, learning rate={learning_rate}"
        )

        return self.data, self.loader, self.model, self.criterion, self.optimizer

    def train_model(self):
        if not pathlib.Path("../data/checkpoints").exists():
            pathlib.Path("../data/checkpoints").mkdir()

        model_file = pathlib.Path("../data/train.pt")

        if not model_file.exists():
            start = time.time()

            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}/{self.epochs}", end=" ")
                start_epoch = time.time()
                running_loss = 0.0

                for i, item in enumerate(self.loader, 0):
                    input = item.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(input)
                    loss = (
                        self.criterion["pitch"](output["pitch"], input[:, 0].long())
                        + self.criterion["step"](
                            output["step"], input[:, 1].view(-1, 1)
                        )
                        + self.criterion["duration"](
                            output["duration"], input[:, 2].view(-1, 1)
                        )
                    )
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                print(f"({time.time() - start_epoch:.2f}s)");
                print(f"  Loss: {running_loss / len(self.loader)}")

                torch.save(self.model.state_dict(), f"../data/checkpoints/epoch_{epoch}.pt")

            print(f"Training Time: {time.time() - start:.2f}s")
            torch.save(self.model.state_dict(), model_file)
        else:
            print("Model already trained")
            self.model.load_state_dict(torch.load(model_file))

        return self.model
    
    def predict_next_note(self, input_notes, temperature=1.0):
        inputs = torch.tensor(input_notes, dtype=torch.float32).unsqueeze(0).to(self.device)
        outputs = self.model(inputs)
        pitch_logits = outputs["pitch"]
        step = outputs["step"]
        duration = outputs["duration"]

        pitch_probs = torch.softmax(pitch_logits / temperature, dim=2)

        pitch = torch.multinomial(pitch_probs[0, -1], 1).item()
        step = step[0, -1].squeeze().item()
        duration = duration[0, -1].squeeze().item()

        return pitch, step, duration
    
    def generate_notes(self, raw_notes, num_predictions, seq_length, temperature=1.0):
        key_order = ["pitch", "step", "duration"]
        sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
        input_notes = sample_notes[:seq_length]

        generated_notes = []
        prev_start = 0
        for _ in range(num_predictions):
            pitch, step, duration = self.predict_next_note(input_notes, temperature)
            start = prev_start + step
            end = start + duration
            input_note = (pitch, step, duration)
            generated_notes.append((*input_note, start, end))
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
            prev_start = start

        generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, "start", "end"))

        out_file = "output.mid"
        out_pm = self.midi.notes_to_midi(
            generated_notes, out_file=out_file, instrument_name="Acoustic Grand Piano"
        )

        return generated_notes, out_pm
