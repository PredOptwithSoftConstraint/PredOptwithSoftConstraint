import torch.nn as nn
import time
import numpy as np
import torch
import math
from config import *
class sp500_element: # the element of the sp500 variant dataset.
    def __init__(self, features, label):
        self.features, self.label = features, label

torch.random.manual_seed(179090813)

def rounding(x, eps):
    if math.fmod(x, 1) < eps: x = math.floor(x)
    elif math.fmod(x, 1) >= 1 - eps: x = math.ceil(x)
    return x

def getconstrlist(x, C, d):
    # print(C.shape, x.shape, d.shape)
    constr = np.matmul(C, x.reshape(-1, 1)) - d.reshape(-1, 1)
    idx_none, idx_linear, idx_quad = [], [], []
    for i in range(C.shape[0]):
        if constr[i] < - 1 / (4 * QUAD_SOFT_K): idx_none.append(i)
        if -1 / (4 * QUAD_SOFT_K) <= constr[i] and constr[i] <= 1 / (4 * QUAD_SOFT_K): idx_quad.append(i)
        if constr[i] > 1 / (4 * QUAD_SOFT_K): idx_linear.append(i)
    return idx_quad, idx_linear, idx_none

def merge_constraints(A0, b0, C0, d0, alpha1, alpha2):
    N, M1, M2 = A0.shape[1], A0.shape[0], C0.shape[0]
    # merged data
    # print(C0.shape, A0.shape)
    C = np.concatenate((C0, -C0, A0, -np.identity(N)), axis=0)
    alpha = np.concatenate((alpha1, alpha2, math.sqrt(N) * 100 * np.ones((M1, 1)), 100 * np.ones((N, 1))), axis=0)
    d = np.concatenate((d0, -d0, b0, np.zeros((N, 1))), axis=0)
    return C, d, alpha

