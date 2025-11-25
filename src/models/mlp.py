import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(512,256,128), dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d,1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits
