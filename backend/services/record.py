import torch
from transformers import Wav2Vec2Model
import torch.nn as nn


class OverlapModel(nn.Module):
    """
    Wav2Vec2-based binary classifier for cross-talk (overlap) detection.
    Loads Wav2Vec2 base encoder strictly from a local directory — no internet.
    """
    def __init__(self, model_dir: str):
        super().__init__()
        if not model_dir:
            raise ValueError("model_dir is required for offline overlap model loading")
        self.encoder = Wav2Vec2Model.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        x = self.encoder(x).last_hidden_state
        x = x.mean(dim=1)
        x = torch.sigmoid(self.fc(x))
        return x
