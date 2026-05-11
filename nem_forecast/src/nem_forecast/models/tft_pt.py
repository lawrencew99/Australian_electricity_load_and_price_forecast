from __future__ import annotations

"""
Temporal Fusion Transformer 薄封装占位。

生产环境建议改用 pytorch-forecasting 的 TemporalFusionTransformer，
或自行实现完整编码器-解码器与变量选择网络。此处仅保留与张量形状兼容的 forward，
便于替换。
"""

import numpy as np
import torch
from torch import nn


class TFTPlaceholder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, horizon: int = 1):
        super().__init__()
        self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) -> (batch, horizon)"""
        out, h = self.encoder(x)
        last = out[:, -1, :]
        return self.fc(last)

    @torch.no_grad()
    def predict_numpy(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        xt = torch.tensor(X, dtype=torch.float32)
        return self.forward(xt).numpy()


class TFTRegressorSklearnLike:
    """与 LSTMRegressor 类似的 sklearn-like 包装（单 horizon 输出）。"""

    def __init__(
        self,
        input_size: int,
        *,
        hidden_size: int = 64,
        horizon: int = 1,
        lr: float = 1e-3,
        max_epochs: int = 5,
        batch_size: int = 64,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self._net = TFTPlaceholder(input_size, hidden_size=hidden_size, horizon=horizon)
        self._opt = torch.optim.Adam(self._net.parameters(), lr=lr)
        self._loss = nn.MSELoss()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.horizon = horizon

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_t = torch.tensor(X, dtype=torch.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
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
                if pred.shape != yb.shape:
                    yb = yb[:, : pred.shape[1]]
                loss = self._loss(pred, yb)
                loss.backward()
                self._opt.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._net.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32)
            pred = self._net(xt).numpy()
        if self.horizon == 1:
            return pred.ravel()
        return pred
