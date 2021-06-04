import numpy as np
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal, HalfNormal, LowRankMultivariateNormal, MultivariateNormal
from torch.distributions import Independent
from extended_distributions import TruncatedNormal
from sklearn.model_selection import train_test_split

dtype = torch.double
device = torch.device("cpu")

torch.set_default_tensor_type('torch.DoubleTensor')


def sparsify_matrix(x, sparsity=0.5):
    if x.ndimension() != 2:
        raise ValueError("Not a matrix: dim %d,  should be 2-dim", x.ndimension())

    rows, cols = x.shape
    num_zeros = int(math.ceil(sparsity * rows))

    for col_idx in range(cols):
        row_indices = torch.randperm(rows)
        zero_indices = row_indices[:num_zeros]
        x[zero_indices, col_idx] = 0
    return x


def SparseUniformMatrix(shape, low, high, sparsity=0.5):  # sparsity表示的是有多少比例的元素会被强制变成0
    x = torch.empty(shape)
    x = nn.init.uniform_(x, low, high)
    x = sparsify_matrix(x, sparsity)
    return x


def SparseTruncNormalMatrix(shape, mean, std, a, b, sparsity=0.5):
    x = torch.empty(shape)
    x = nn.init.trunc_normal_(x, mean, std, a, b)
    x = sparsify_matrix(x, sparsity)
    return x


def SparseNormalMatrix(shape, mean, std, sparsity=0.5):
    x = torch.empty(shape)
    x = nn.init.normal_(x, mean, std)
    x = sparsify_matrix(x, sparsity)
    return x

def lowRankSynthesize(n, input_dim, output_dim, latent_dim, feature_rank, net_sparsity=0.5, noise_level=0.02, seed=123):
    """ Generate feaftures and labels with a latent random variable.
        This is more close to reality.
    """
    np.random.seed(seed)
    torch.manual_seed(seed * 31 % (2 ** 32))

    class Project(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(Project, self).__init__()
            self.A = Variable(torch.randn((input_dim, output_dim),
                                          device=device, dtype=dtype, requires_grad=False))
            self.b = Variable(torch.randn((output_dim),
                                          device=device, dtype=dtype, requires_grad=False))
            nn.init.sparse_(self.A, sparsity=net_sparsity)
            nn.init.normal_(self.b, 0, 0.1)

        def forward(self, x):
            z = torch.matmul(x, self.A) + self.b
            # z = torch.sigmoid(z)
            z = torch.relu(z)
            # z = torch.exp(z)
            return z

    def sparse_init(m):
        if type(m) == nn.Linear:
            nn.init.sparse_(m.weight, sparsity=net_sparsity)
            nn.init.normal_(m.bias, 0, 0.1)

    class MLP(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(MLP, self).__init__()
            self.net = nn.Sequential(nn.Linear(input_dim, math.ceil(1.5 * input_dim)),
                                     nn.ReLU(),
                                     nn.Linear(math.ceil(1.5 * input_dim), 128),
                                     nn.ReLU(),
                                     nn.Linear(128, output_dim))
            self.net.apply(sparse_init)

        def forward(self, x):
            with torch.no_grad():
                z = self.net(x)
            return z

    # X_true
    x_loc = torch.rand(input_dim)
    x_cov_factor = torch.rand(input_dim, feature_rank)
    # x_cov_diag = torch.diag(torch.ones(input_dim))
    x_cov_diag = torch.ones(input_dim)
    x_m = LowRankMultivariateNormal(x_loc, x_cov_factor, x_cov_diag)
    X_true = x_m.rsample([n])

    B = torch.from_numpy(np.random.randint(2, size=(input_dim, output_dim))).double()
    z = torch.sin(2 * math.pi * (X_true @ B))
    y_true = MLP(output_dim, output_dim)(z)

    avgy, miny, maxv = torch.zeros(output_dim), torch.zeros(output_dim), torch.zeros(output_dim)
    for j in range(output_dim):
        avgy[j] = torch.mean(y_true[:, j])
        miny[j] = torch.min(y_true[:, j])
        maxv[j] = torch.max(y_true[:, j]) - miny[j]
    ratio = torch.zeros(n, output_dim)
    importance = torch.floor(SparseUniformMatrix(shape=(output_dim, 1), low=0, high=2,
                            sparsity=0)+0.5)
    for i in range(n):
        for j in range(output_dim):
            ratio[i, j] = (y_true[i, j] - miny[j]) / maxv[j]
            y_true[i, j] = ratio[i, j]
    #print("z:", z)
    y_tns = TruncatedNormal(loc=torch.empty(output_dim).fill_(0), scale=torch.ones(output_dim) * noise_level,
                            a=torch.tensor(output_dim).fill_(0), b=torch.tensor(output_dim).fill_(1.5))
    # y_tns = TruncatedNormal(loc=torch.rand(output_dim)*10, scale=torch.ones(output_dim)*1,
    #                        a=torch.tensor(output_dim).fill_(0), b=torch.tensor(output_dim).fill_(1.5))
    y_noise_m = Independent(y_tns, 1)
    y = y_true + y_noise_m.rsample([n])
    # y = y * torch.rand(output_dim) * 100 #FIXME

    # X
    # x_noise_scale_tril = torch.diag(torch.ones(input_dim)*noise_level)
    x_noise_scale_tril = torch.diag(nn.init.trunc_normal_(torch.empty(input_dim), 0.5, 0.1, 0, 2))
    x_noise_m = MultivariateNormal(torch.zeros(input_dim), scale_tril=x_noise_scale_tril)
    X = X_true + x_noise_m.rsample([n])
    """
    print("noisesample:", y_noise_m.rsample([n]))
    print("noise:", noise_level)
    """
    y, X = y_true, X_true
    # test will round up in train_test_split
    # train = 100, valid = 25, test = 25 
    # train = 1000, valid = 250, test = 250
    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.166, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state=seed)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), importance

def generateProblemParams(theta, dim_hard, dim_soft, importance, A_sparsity=0.5, C_sparsity=0.5,
                          b_ratio=0.5, d_ratio=0.25, seed=2021):
    """generate constraints
       main utility: <theta, x>
       soft constraints(punishment): <alpha, max(Cx-d, 0)>
       hard constraints: Ax <= b, x >= 0
    """

    torch.manual_seed(seed * 31 % (2 ** 32))

    A = SparseUniformMatrix(shape=(dim_hard, theta.shape[-1]), low=0, high=1,
                            sparsity=A_sparsity)
    b = A @ torch.ones((theta.shape[-1], 1)) * b_ratio 
    for i in range(theta.shape[-1]):
        A[:, i] *= (1 ** importance[i])
    C = SparseUniformMatrix(shape=(dim_soft, theta.shape[-1]), low=0, high=1,
                            sparsity=C_sparsity)
    d = C @ torch.ones((theta.shape[-1], 1)) * d_ratio 
    for i in range(theta.shape[-1]):
        C[:, i] *= (1 ** importance[i]) # importance is deprecated.
    alpha = nn.init.uniform_(torch.empty(dim_soft, 1), 0, 0.2)
    if alpha.shape[0] == 1: # disable the soft constraint
        print("the soft constraint is banned!")
        alpha[0] = 0
    return A.numpy(), b.numpy(), C.numpy(), d.numpy(), alpha.numpy()

