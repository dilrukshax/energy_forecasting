"""Compact recurrent regressors and causal sequence sampling."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Return a sequence ending at feature row t and its target y[t].

    F[t] itself contains only previous measurements and known calendar values.
    Windows are views created on demand, avoiding a large copied 3D array.
    """
    def __init__(self, features, targets, target_indices, lookback):
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.indices = np.asarray(target_indices, dtype=int)
        self.lookback = lookback
        if len(self.indices) and self.indices.min() < lookback - 1:
            raise ValueError("Insufficient history for sequence.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        t = self.indices[item]
        return torch.from_numpy(self.features[t-self.lookback+1:t+1]), torch.tensor(self.targets[t])


class EnergyRNN(nn.Module):
    """LSTM/GRU -> dropout -> dense ReLU -> scalar linear regression output."""
    def __init__(self, n_features, kind="LSTM", hidden=32, layers=1, dropout=0.1):
        super().__init__()
        if kind not in {"LSTM", "GRU"}:
            raise ValueError(f"Unsupported architecture: {kind}")
        cell = nn.LSTM if kind == "LSTM" else nn.GRU
        self.rnn = cell(n_features, hidden, num_layers=layers, batch_first=True,
                        dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, sequence):
        states, _ = self.rnn(sequence)
        return self.head(states[:, -1, :]).squeeze(-1)
