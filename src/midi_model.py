import pathlib
import time
import zipfile
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import pretty_midi
from torch.utils.data import DataLoader
from midi_data import MidiData
from model import LSTMModel
from midi_handler import MidiHandler


class MidiModel:
    def __init__(self):
        r"""
        This class is used to handle the model and the training data.
        """

        self.midi = MidiHandler()

        self.device = "cpu"
        self.epochs = 50
        self.batch_size = 64
        self.learning_rate = 0.0005

        self.input_size = 128
        self.hidden_size = 256
        self.output_size = 128

        self.pitch_weight = 1.0
        self.step_weight = 1.0
        self.duration_weight = 1.0

        self.train_notes = None
        self.data = None
        self.loader = None
        self.model = None
        self.criterion = None
        self.optimizer = None

        self.output_dir = pathlib.Path("../data")
        self.checkpoints_dir = pathlib.Path(self.output_dir / "checkpoints")
        self.notes_dir = pathlib.Path(self.output_dir / "notes")
        self.maestro_dir = pathlib.Path(self.output_dir / "maestro-v2.0.0")
        self.generated_dir = pathlib.Path(self.output_dir / "generated")

    ####################################################################################################################

    def init_data_dir(self):
        r"""
        Initializes the data directory.
        """

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)

    ####################################################################################################################

    def load_train_notes(self, range_from: int = 0, range_to: int | None = None):
        r"""
        Loads notes from samples in a given range. Saves the notes to a file for future use: `data/notes/range-{range_from}-{range_to}.pt`.

        If the file already exists, the notes are loaded from the file.
        """

        if range_to is None:
            range_to = len(self.midi.files)

        notes_file = pathlib.Path(self.notes_dir / f"range-{range_from}-{range_to}.pt")

        if not notes_file.exists():
            print("Parsing notes from samples")

            all_notes = self.midi.get_notes_from_range(range_from, range_to)
            key_order = ["pitch", "step", "duration"]
            self.train_notes = torch.tensor(all_notes[key_order].values, dtype=torch.float32).to(self.device)
            torch.save(self.train_notes, notes_file)

        else:
            print("Loading notes from file")
            self.train_notes = torch.load(notes_file)

        print("Number of notes parsed:", len(self.train_notes))

    ####################################################################################################################

    def create_model(self):
        r"""
        Creates the model with the given parameters.
        """

        self.data = MidiData(self.train_notes)
        self.loader = DataLoader(self.data, batch_size=self.batch_size)

        self.model = LSTMModel(self.input_size, self.hidden_size, self.output_size)
        self.model.to(self.device)

        self.criterion = {
            "pitch": nn.CrossEntropyLoss(),
            "step": nn.MSELoss(),
            "duration": nn.MSELoss(),
        }
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(f"Training Model: epochs={self.epochs}, batch size={self.batch_size}, learning rate={self.learning_rate}")

    ####################################################################################################################

    def train_model(self):
        r"""
        Trains the model with the given data and parameters. Saves the model to a file for future use: `data/train.pt`.

        Checkpoints are saved after each epoch: `data/checkpoints/epoch_{epoch}.pt`.

        If the model is already trained, it is loaded from the file.

        TODO - Add a way to save multiple models based on range, epoch, batch size, and learning rate.
        """

        model_file = pathlib.Path(self.output_dir / "train.pt")

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

                    pitch_loss = self.criterion["pitch"](output["pitch"], input[:, 0].long()) * self.pitch_weight
                    step_loss = self.criterion["step"](output["step"], input[:, 1].view(-1, 1)) * self.step_weight
                    duration_loss = self.criterion["duration"](output["duration"], input[:, 2].view(-1, 1)) * self.duration_weight
                    loss = pitch_loss + step_loss + duration_loss

                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                print(f"({time.time() - start_epoch:.2f}s)")
                print(f"  Loss: {running_loss / len(self.loader)}")

                torch.save(self.model.state_dict(), self.output_dir / f"checkpoints/epoch_{epoch}.pt")

            print(f"Training Time: {time.time() - start:.2f}s")
            torch.save(self.model.state_dict(), model_file)

        else:
            print("Model already trained")
            self.model.load_state_dict(torch.load(model_file))

    ####################################################################################################################

    def predict_next_note(self, input_notes, temperature=1.0):
        r"""
        Takes initial input notes and predicts the next note based on the model.
        """

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

    ####################################################################################################################

    def generate_notes(self, raw_notes, num_predictions: int, seq_length: int, temperature: float = 1.0):
        r"""
        Generates notes based on the given input notes. Saves the generated notes to the file: `data/generated/output.mid`.
        """

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

        outfile = str(self.generated_dir / "output.mid")
        out_pm = self.midi.notes_to_midi(generated_notes, outfile, pretty_midi.constants.INSTRUMENT_MAP[0])

        return generated_notes, out_pm

    ####################################################################################################################

    def load_train_data(self):
        r"""
        TODO
        """

        if not self.maestro_dir.exists():
            print("Downloading Maestro Dataset")

            torch.hub.download_url_to_file(
                "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip",
                self.output_dir / "maestro-v2.0.0-midi.zip",
            )

            with zipfile.ZipFile(self.output_dir / "maestro-v2.0.0-midi.zip", "r") as zip_ref:
                zip_ref.extractall(self.output_dir)
                pathlib.Path(self.output_dir / "maestro-v2.0.0-midi.zip").unlink()

        else:
            print("Maestro Dataset already downloaded")

        self.midi.load_files(str(self.maestro_dir / "**/*.mid*"))
