import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class DataHandler(Dataset):
    def __init__(self, X):
        self.X = X

        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.len


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
            "pitch": pitch_out,
            "step": step_out,
            "duration": duration_out,
        }
