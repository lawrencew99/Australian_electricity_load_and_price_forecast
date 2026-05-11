from __future__ import annotations

import numpy as np
import torch
from torch import nn


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


class LSTMRegressor:
    """
    占位 LSTM：输入形状 (batch, seq_len, n_features)，回归单步目标。
    """

    def __init__(
        self,
        input_size: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        max_epochs: int = 5,
        batch_size: int = 64,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.input_size = input_size
        self._net = _LSTMNet(input_size, hidden_size, num_layers, dropout)
        self._opt = torch.optim.Adam(self._net.parameters(), lr=lr)
        self._loss = nn.MSELoss()
        self.max_epochs = max_epochs
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """X: (N, seq_len, input_size)"""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        n = X_t.shape[0]
        self._net.train()
        for _ in range(self.max_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = X_t[idx]
                yb = y_t[idx]
                self._opt.zero_grad()
                pred = self._net(xb)
                loss = self._loss(pred, yb)
                loss.backward()
                self._opt.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._net.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32)
            pred = self._net(xt).numpy()
        return pred


def numpy_to_lstm_sequences(
    X_flat: np.ndarray,
    n_lags: int,
    n_extra: int,
) -> np.ndarray:
    """
    将 (N, n_lags + n_extra) 展平特征转为 (N, seq_len, input_size)。
    此处 seq_len=n_lags，每步单变量滞后展开为每时刻 1 维；额外特征复制到每一步末尾拼接简化：
    实际做法：最后一步拼接 extra（形状 N, n_lags + extra）。
    """
    # 简化：seq 维度取 lag 序列每步 1 维 + 在最后一步附带 extras 的平均——脚手架用最后一行拼接 extras
    lags = X_flat[:, :n_lags].reshape(X_flat.shape[0], n_lags, 1)
    if n_extra <= 0:
        return lags
    extras = X_flat[:, n_lags:]
    extra_pad = np.repeat(extras[:, np.newaxis, :], n_lags, axis=1)
    return np.concatenate([lags, extra_pad], axis=-1)
