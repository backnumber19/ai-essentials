__all__ = ["LSTMModel", "generate_nonstationary_data"]

import torch
import torch.nn as nn
import numpy as np


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode="norm"):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / self.affine_weight
            x = x * self.stdev + self.mean
            return x


class LSTMModel(nn.Module):
    def __init__(
        self, input_size=7, hidden_size=64, num_layers=2, pred_len=24, use_revin=True
    ):
        super().__init__()
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(input_size)

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.1
        )
        self.fc = nn.Linear(hidden_size, input_size)
        self.pred_len = pred_len

    def forward(self, x):
        if self.use_revin:
            x = self.revin(x, mode="norm")
        lstm_out, _ = self.lstm(x)
        pred = self.fc(lstm_out[:, -self.pred_len :, :])
        if self.use_revin:
            pred = self.revin(pred, mode="denorm")
        return pred


def generate_nonstationary_data(
    n_samples, seq_len, n_features, mean_range, std_range, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    data = []
    for _ in range(n_samples):
        mean = np.random.uniform(*mean_range)
        std = np.random.uniform(*std_range)
        trend = np.linspace(0, np.random.uniform(-10, 10), seq_len)
        series = np.random.randn(seq_len, n_features) * std + mean
        series += trend[:, np.newaxis]
        data.append(series)

    return torch.FloatTensor(np.array(data))