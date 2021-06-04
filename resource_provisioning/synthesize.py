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
# device = torch.device("cuda:0") # Uncomment this to run on GPU

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

def lowRankSynthesize(n, m1, input_dim, output_dim, feature_rank, net_sparsity=0.5, noise_level=0.02, seed=123):
    """ Generate feaftures and labels with a latent random variable.
        This is more close to reality.
    """
    # M1 is soft constraint.
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
            z = torch.relu(z)
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
                                     nn.Linear(math.ceil(1.5 * input_dim), 256),
                                     nn.ReLU(),
                                     nn.Linear(256, output_dim))
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

    y_true = torch.abs(MLP(input_dim, output_dim * m1)(X_true))

    avgy, miny, maxv = torch.zeros(output_dim * m1), torch.zeros(output_dim * m1), torch.zeros(output_dim * m1)

    if not (n == 2 and output_dim == 1): # debug mode off.
        for j in range(output_dim * m1):
            avgy[j] = torch.mean(y_true[:, j])
            miny[j] = torch.min(y_true[:, j])
            maxv[j] = torch.max(y_true[:, j]) - miny[j]

        for i in range(n):
            for j in range(output_dim * m1):
                y_true[i, j] = (y_true[i, j] - miny[j]) / maxv[j]
    y_tns = TruncatedNormal(loc=torch.empty(output_dim * m1).fill_(0), scale=torch.ones(output_dim * m1) * noise_level,
                            a=torch.tensor(output_dim * m1).fill_(0), b=torch.tensor(output_dim * m1).fill_(1.5))
    y_noise_m = Independent(y_tns, 1)
    y = y_true + y_noise_m.rsample([n])
    # y = y * torch.rand(output_dim) * 100 #FIXME

    # X
    x_noise_scale_tril = torch.diag(nn.init.trunc_normal_(torch.empty(input_dim), 0.5, 0.1, 0, 2))
    x_noise_m = MultivariateNormal(torch.zeros(input_dim), scale_tril=x_noise_scale_tril)
    X = X_true + x_noise_m.rsample([n])

    y, X = y_true, X_true
    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=seed)
    if n == 2:
        if output_dim in [2, 3]: # debug mode
            y_train = torch.eye(output_dim).unsqueeze(0).numpy()
        elif output_dim == 1:
            y_train = np.array([[[3]]])
    y_train = y_train.reshape(y_train.shape[0], m1, output_dim)  # labels are matrices of C.
    y_test = y_test.reshape(y_test.shape[0], m1, output_dim)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    return (X_train, y_train), (X_test, y_test)


def generateProblemParams(dim_context, dim_hard, dim_soft, A_sparsity=0, # TODO: this should be 0.5!
                          b_ratio=0.5, d_ratio=0.25, seed=2021):
    """generate constraints
       main utility: <theta, x>
       soft constraints(punishment): <alpha, max(Cx-d, 0)>
       hard constraints: Ax <= b, x >= 0
    """

    torch.manual_seed(seed * 31 % (2 ** 32))

    #A = SparseUniformMatrix(shape=(dim_hard, dim_context), low=0, high=1, sparsity=A_sparsity)
    A = torch.cat((torch.ones(1, dim_soft), -torch.ones(1, dim_soft)), dim=0)
    b = torch.tensor([[1], [-1]]) # x1+...+xn = 1
    d = torch.clamp(torch.ones(dim_context, 1) * 0.5 + 0.1 * torch.randn(size=(dim_context, 1)), min=0) # TODO: this will be tuned later.
    alpha1 = torch.ones(dim_context, 1) * 50
    alpha2 = torch.ones(dim_context, 1) * 0.5
    return A.numpy(), b.numpy(), d.numpy(), alpha1.numpy(), alpha2.numpy()

