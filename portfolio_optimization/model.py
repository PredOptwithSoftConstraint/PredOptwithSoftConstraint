import torch
import torch.nn as nn
from utils import computeCovariance

def linear_block(in_channels, out_channels, seed, activation='ReLU'):
    torch.manual_seed(seed)
    if activation == 'ReLU':
        return nn.Sequential(
               nn.Linear(in_channels, out_channels),
               nn.BatchNorm1d(out_channels),
               # torch.nn.Dropout(p=0.5),
               nn.LeakyReLU()
               )
    elif activation == 'Sigmoid':
        return nn.Sequential(
               nn.Linear(in_channels, out_channels),
               nn.BatchNorm1d(out_channels),
               # torch.nn.Dropout(p=0.5),
               nn.Sigmoid()
               )

class PortfolioModel(nn.Module):
    def __init__(self, seed, input_size=20, output_size=1):
        # input:  features
        # output: embedding
        super(PortfolioModel, self).__init__()
        torch.manual_seed(seed)
        self.input_size, self.output_size = input_size, output_size
        self.model = nn.Sequential(
                linear_block(input_size, 100, seed),
                linear_block(100, 100, seed),
                linear_block(100, output_size, seed, activation='Sigmoid')
                ).double()

    def forward(self, x):
        y = self.model(x)
        return (y - 0.5) * 0.1

class CovarianceModel(nn.Module):
    def __init__(self, n, seed):
        super(CovarianceModel, self).__init__()
        torch.manual_seed(seed)
        self.n = n
        self.latent_dim = 32
        self.embedding = nn.Embedding(num_embeddings=self.n, embedding_dim=self.latent_dim).double()

    def forward(self):
        security_embeddings = self.embedding(torch.LongTensor(range(self.n)))
        cov = computeCovariance(security_embeddings)
        return cov


