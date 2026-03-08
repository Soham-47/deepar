import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepAR(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_targets: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.1
                 ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_targets = n_targets

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        self.linear = nn.Linear(self.hidden_size, 2 * self.n_targets)

    def forward(self, x):
        out, _ = self.lstm(x)
        outputs = self.linear(out)
        mu = outputs[:, :, :self.n_targets]
        sigma = outputs[:, :, self.n_targets:]
        sigma = F.softplus(sigma) + 1e-6
        return mu, sigma
