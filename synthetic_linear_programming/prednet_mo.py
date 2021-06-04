import torch
import torch.nn as nn
from config import *

class PredNetMO(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PredNetMO, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
            ).double()
    def forward(self, x):
        return self.net(x)

