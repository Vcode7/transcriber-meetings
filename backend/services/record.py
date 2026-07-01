import torch
from transformers import Wav2Vec2Model
import torch.nn as nn

class OverlapModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        x = self.encoder(x).last_hidden_state
        x = x.mean(dim=1)
        x = torch.sigmoid(self.fc(x))
        return x