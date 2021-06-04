import torch
import torch.nn as nn
from config import *

class PredNet3D(nn.Module):
    def __init__(self, input_shape, output_shape, seed):
        super(PredNet3D, self).__init__()
        # input_shape = [*, 24, 77]
        # output_shape = [*, 24]
        dim1 = input_shape[-2]
        dim2 = input_shape[-1]
        dim3 = output_shape[-1]
        torch.manual_seed(seed)
        self.net1 = nn.Sequential(
            nn.Linear(dim2, dim2),
            nn.ReLU(),
            nn.Linear(dim2, dim2),
            nn.ReLU(),
            nn.Linear(dim2, dim2),
            nn.ReLU(),
            nn.Linear(dim2, dim2),
            nn.ReLU(),
            nn.Linear(dim2, dim2),
            nn.ReLU()).double()
        self.net2 = nn.Sequential(
            nn.Linear(dim1*dim2, dim1*dim2//2),
            nn.ReLU(),
            nn.Linear(dim1*dim2//2, dim3)
            ).double()

    def forward(self, x):
        # print("shape:", x.shape)
        x1 = self.net1(x)
        x2 = x1.view(x1.shape[0], -1)
        return self.net2(x2)
