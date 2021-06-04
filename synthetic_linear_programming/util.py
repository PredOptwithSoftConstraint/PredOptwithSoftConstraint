import torch.nn as nn
import time
import numpy as np
import torch
import math
from config import *
class sp500_element: # the element of the sp500 variant dataset.
    def __init__(self, features, label):
        self.features, self.label = features, label

class dataset_element: # the element of the trivial dataset.
    def __init__(self, beta_1, beta_2):
        self.beta_1, self.beta_2 = beta_1, beta_2
        self.features, self.label = calc_features(beta_1, beta_2)
torch.random.manual_seed(179090813)
twistnet1 = nn.Sequential( # deprecated.
    nn.Linear(28, 28),
    nn.Sigmoid(),
    nn.Linear(28, 28),
    nn.Sigmoid(),
    nn.Linear(28, 28),
    nn.Sigmoid(),
    nn.Linear(28, 28),
    nn.Tanh(),
    nn.Linear(28, 1)
).double()
for param in twistnet1.parameters():
    param.requires_grad = False

def Synthesize_from_SP500_debug(train_dataset):
    trainset, validset, testset = [], [], []
    trainlen, validlen, testlen = 1400, 400, 200
    for i in range(trainlen):
        idx = train_dataset.sampler.indices[i]
        trainset.append(sp500_element(train_dataset.dataset[idx][0], torch.clamp(train_dataset.dataset[idx][2], min=-0.1, max=0.1) * 10 + 1))
    for i in range(trainlen, trainlen + validlen):
        idx = train_dataset.sampler.indices[i]
        validset.append(sp500_element(train_dataset.dataset[idx][0], torch.clamp(train_dataset.dataset[idx][2], min=-0.1, max=0.1) * 10 + 1))
    for i in range(trainlen + validlen, trainlen + validlen + testlen):
        idx = train_dataset.sampler.indices[i]
        testset.append(sp500_element(train_dataset.dataset[idx][0], torch.clamp(train_dataset.dataset[idx][2], min=-0.1, max=0.1) * 10 + 1))
    return trainset, len(trainset), validset, len(validset), testset, len(testset)

def Twist(A): # deprecated.
    n, m = A.shape[0], A.shape[1]
    twistmatrix = np.random.randint(1, 4, size=(n, n)) / 2
    for i in range(n):
        for j in range(n):
            if i != j: twistmatrix[i, j] = 0
    return np.matmul(twistmatrix, A)

def rounding(x, eps):
    if math.fmod(x, 1) < eps: x = math.floor(x)
    elif math.fmod(x, 1) >= 1 - eps: x = math.ceil(x)
    return x

def calc_features(beta_1, beta_2): # beta1 and beta2 are 1-dimensional np arrays; deprecated.
    features = np.zeros((len(beta_1), 28))
    label = np.zeros((len(beta_1), 1))

    for i in range(29): # last 28 days
        if i < 28: features[:, i] = (np.abs(np.sin(beta_1 + i)) * 3 + np.abs(np.cos(beta_2 + i)) * 5 + np.random.normal(size=1) * 0.05).squeeze()
        else: label = np.abs(np.sin(beta_1 + i)) * 3 + np.abs(np.cos(beta_2 + i)) * 5 + np.random.normal(size=1) * 0.05
    features, label = twistnet1(torch.from_numpy(features)), torch.from_numpy(label)
    return features, label

def getconstrlist(x, C, d):
    # print(C.shape, x.shape, d.shape)
    constr = np.matmul(C, x.reshape(-1, 1)) - d.reshape(-1, 1)
    idx_none, idx_linear, idx_quad = [], [], []
    for i in range(C.shape[0]):
        if constr[i] < - 1 / (4 * QUAD_SOFT_K): idx_none.append(i)
        if -1 / (4 * QUAD_SOFT_K) <= constr[i] and constr[i] <= 1 / (4 * QUAD_SOFT_K): idx_quad.append(i)
        if constr[i] > 1 / (4 * QUAD_SOFT_K): idx_linear.append(i)
    return idx_quad, idx_linear, idx_none

def merge_constraints(A0, b0, C0, d0, alpha0, theta):
    N, M1, M2 = A0.shape[1], A0.shape[0], C0.shape[0]
    # merged data
    C = np.concatenate((C0, A0, -np.identity(N)), axis=0)
    alpha = np.concatenate((alpha0, math.sqrt(N) * 5 * np.max(np.abs(theta)) * np.ones((M1, 1)), np.zeros((N, 1))), axis=0)
    d = np.concatenate((d0, b0, np.zeros((N, 1))), axis=0)
    # construct alpha
    """
    mxratio = 0
    # Deprecated; no need to calculate the angle actually.
    for i in range(C.shape[0]):  
        n = np.zeros((C.shape[1], 1))  
        A = np.zeros((C.shape[1] - 1, C.shape[1]))
        for j in range(C.shape[1] - 1): 
            x1, x2 = np.random.random((C.shape[1], 1)), np.random.random((C.shape[1], 1))
            k = 0
            for k2 in range(C.shape[1]):
                k = k2
                if C[i, k2] != 0: break
            # k is the non-zero part
            x1[k], x2[k] = x1[k] - (C[i].dot(x1) - d[i]) / C[i, k], x2[k] - (C[i].dot(x2) - d[i]) / C[i, k]
            A[j] = (x1 - x2).squeeze()
        j = 0
        for j2 in range(C.shape[1]):
            j = j2
            if C[i, j2] != 0: break
        # j is the non-zero part
        n[j] = 1
        idx, b_solve = [i for i in range(j)] + [i for i in range(j + 1, A.shape[1])], np.zeros((A.shape[0]))
        for l in range(A.shape[0]): b_solve[l] = -A[l, j]
        # print(A[0], C[i].dot(x1) - d[i], C[i].dot(x2) - d[i])
        if j == 0:
            A_solve = A[:, 1:]
        elif j == A.shape[1] - 1:
            A_solve = A[:, :A.shape[1] - 1]
        else:
            A_solve = np.concatenate((A[:, :j], A[:, j + 1:]), axis=1)
        # print(A)
        # print(np.linalg.matrix_rank(A_solve), A.shape, j, A.shape[0]-1)
        # print("C:",C,"d:",d,"A:", A)
        # print(A_solve)
        nn = np.matmul(np.linalg.inv(A_solve), b_solve)
        # print("nn:", nn)
        for l in range(A.shape[0]): n[idx[l]] = nn[l]
        # print("C:",C[i],"d:", d[i], "n:", n)
        ln_n = math.sqrt(n.T.dot(n))
        # cos_n = np.min(n) / ln_n
        cos_n = 1
        eps = 1e-8
        for i in range(n.shape[0]):
            if n[i] > eps: cos_n = min(cos_n, n[i] / ln_n)
            # print(i,":", n[i] ,"/", ln_n)
        if cos_n < 1 - eps: mxratio = max(mxratio, 1 / cos_n)
    print("mxratio:", mxratio)
    exit(0)
    """
    alpha[M1 + M2: M1 + M2 + N] = 20 * np.ones((theta.shape[0], 1)) # math.sqrt(mxratio ** 2 + 1) * np.max(np.abs(theta))

    return C, d, alpha
