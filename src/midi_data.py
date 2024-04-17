from torch.utils.data import Dataset


class MidiData(Dataset):
    def __init__(self, X):
        self.X = X
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.len
