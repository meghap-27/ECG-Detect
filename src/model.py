"""1D CNN for heartbeat classification.

The convolutional branch learns beat morphology; the RR-interval features are
concatenated before the classifier head, because timing (e.g. a premature beat)
is what separates many S beats from N beats, not shape alone.
"""
import torch
from torch import nn


def conv_block(c_in, c_out, k):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, padding=k // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
    )


class ECGNet(nn.Module):
    def __init__(self, n_classes=5, n_rr=4, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32, 7),
            conv_block(32, 64, 5),
            conv_block(64, 128, 5),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + n_rr, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, beat, rr):
        # beat: (batch, length) or (batch, 1, length); rr: (batch, n_rr)
        if beat.dim() == 2:
            beat = beat.unsqueeze(1)
        z = self.features(beat).squeeze(-1)
        return self.head(torch.cat([z, rr], dim=1))
