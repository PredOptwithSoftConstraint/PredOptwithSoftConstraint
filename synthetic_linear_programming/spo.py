import numpy as np
from config import *
import gurobipy
from scipy.optimize import minimize, LinearConstraint
from util import *
from torch.optim import SGD
from torch.autograd import Variable
import gurobipy as gp
from gurobipy import GRB
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class spo_loss(torch.autograd.Function): # with gradient.
    @staticmethod
    def forward(ctx, *args):
        thetatensor, ground_truth_theta, alpha0, A, b = args
        c = -torch.cat((ground_truth_theta.view(-1, 1), -alpha0)).detach().to(device)
        c_hat = -torch.cat((thetatensor.view(-1, 1), -alpha0)).detach().to(device)
        w1 = torch.from_numpy(lin_getopt(c.cpu().numpy(), A.cpu().numpy(), b.cpu().numpy())).to(device)  # w^*(c)
        w2 = torch.from_numpy(lin_getopt((2 * c_hat - c).cpu().numpy(), A.cpu().numpy(), b.cpu().numpy())).to(device)  # w^*(2 * c_hat - c)
        v1 = lin_getval(ground_truth_theta, alpha0, c).detach()  # z^*(c)
        v2 = lin_getval(ground_truth_theta, alpha0, (2 * c_hat - c)).detach()  # z^*(2 * c_hat - c) = eta_s(c - 2 * c_hat)
        result = v2 + 2 * c_hat.t() @ w1 - v1
        ctx.save_for_backward(w1, w2, c_hat, v1, v2, thetatensor)
        return result

    @staticmethod
    def backward(ctx, *grad_output):
        torch.set_printoptions(precision=8) # for debugging.
        w1, w2, c_hat, v1, v2, thetatensor = ctx.saved_tensors
        # print("w1:", w1.shape, "w2:", w2.shape, "thetatensor:", thetatensor.shape)
        grd = grad_output[0] * 2 *(w1 - w2)[:thetatensor.shape[0], :]
        return grd.view(-1), None, None, None, None

def lin_getopt(theta, A, b):
    ev = gp.Env(empty=True)
    ev.setParam('OutputFlag', 0)
    ev.start()
    m = gp.Model("matrix1", env=ev)
    x = m.addMVar(shape=theta.shape[0], vtype=GRB.CONTINUOUS, name='x')
    m.setObjective(theta.T @ x, GRB.MINIMIZE) # Das ist feasible & unbounded.
    # m.setObjective(0, GRB.MAXIMIZE)
    m.addConstr(A @ x <= b.squeeze(), name='c4')
    m.addConstr(x >= 0, name="c3")
    m.optimize()
    return x.getAttr('x').reshape(-1, 1)

def getopt_spo(ground_truth_theta, thetatensor, alpha0, A, b): # the surrogate function
    c = -torch.cat((torch.from_numpy(ground_truth_theta).view(-1, 1), -torch.from_numpy(alpha0))).detach().to(device)
    c_hat = -torch.cat((thetatensor.view(-1, 1), -torch.from_numpy(alpha0).to(device))).detach().to(device)
    # problem: max (\theta -\alpha)^T(x z), which requires negative c

    # z: (d.shape[0], 1)
    # ( A   0 )    ( b )
    # ( -I  0 ) <= ( 0 )
    # ( 0  -I )    ( 0 )
    # ( C  -I )    ( d )
    # print("thetatensor:", thetatensor)
    grd = spo_loss.apply(thetatensor, torch.from_numpy(ground_truth_theta).to(device), torch.from_numpy(alpha0).to(device), A, b)
    # print("solved eks:",x.reshape(1, -1))
    val = lin_getval(torch.from_numpy(ground_truth_theta).to(device), torch.from_numpy(alpha0).to(device), torch.from_numpy(lin_getopt(c_hat.cpu().numpy(), A.cpu().numpy(), b.cpu().numpy())).to(device))
    return val, grd  # grd is the objective value with real theta, while val is the objective value with predicted theta.

def lin_getval(theta, alpha, x):
    #print(theta.shape, alpha.shape, x.shape)
    return torch.cat((theta.view(-1, 1), -alpha)).t() @ x
