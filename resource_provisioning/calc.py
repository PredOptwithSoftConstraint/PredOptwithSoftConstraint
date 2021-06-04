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


class quad_surro_tensor(torch.autograd.Function): # with gradient.
    @staticmethod
    def forward(ctx, *args):
        Ctensor, Ctrue, inver, diaga, alpha, C, d, delta, gamma = args
        # This matrix C is from Ctensor.
        # Ctrue is merged from label.
        # note: this delta and diaga is not determined from predicted C, but from real C!
        inver, diaga, alpha, C, d, delta, gamma = torch.from_numpy(inver).double().to(device), diaga.double().to(device), torch.from_numpy(alpha).double().to(device), torch.from_numpy(C).double().to(device), torch.from_numpy(d).double().to(device), delta.double().to(device), gamma.double().to(device)
        Ctrue = torch.from_numpy(Ctrue).double().to(device)
        # x = (C^T * delta * diaga * C)^(-1) * (theta + C^T * delta * diaga * d - C^T * (1/4k * delta + gamma) * alpha)
        x = inver @ (C.t() @ delta @ diaga @ d - C.t() @ (1.0 / (4 * QUAD_SOFT_K) * delta + gamma) @ alpha)
        z = Ctrue @ x - d
        result = - alpha.t() @ (delta / 2.0 @ ((z + 1.0 / (4.0 * QUAD_SOFT_K)) ** 2) + gamma @ z)
        ctx.save_for_backward(Ctensor, Ctrue, inver, diaga, alpha, C, d, delta, gamma, x)
        return result

    @staticmethod
    def backward(ctx, *grad_output):
        t1 = time.time()
        torch.set_printoptions(precision=8) # for debugging.
        Ctensor, Ctrue, inver, diaga, alpha, C, d, delta, gamma, x = ctx.saved_tensors
        z_true = Ctrue @ x - d
        df_dC = torch.zeros(C.shape)
        # dx_dC = torch.zeros(alpha1.shape[0], C.shape[0], C.shape[1])
        eta = delta @ diaga @ d - (delta / (4 * QUAD_SOFT_K) + gamma) @ alpha
        beta = C.t() @ delta @ diaga @ d - C.t() @ (delta / (4 * QUAD_SOFT_K) + gamma) @ alpha
        assert C.shape[0] == eta.shape[0], "Shape Mismatch!"

        true_delta, true_gamma = np.zeros_like(delta), np.zeros_like(gamma)
        for i in range(delta.shape[0]):
            if z_true[i] >= 1 / (4 * QUAD_SOFT_K):
                true_gamma[i, i] = 1
            elif z_true[i] >= -1 / (4 * QUAD_SOFT_K):
                true_delta[i, i] = 2 * QUAD_SOFT_K

        df_dx = - Ctrue.t() @ true_delta @ diaga @ z_true - Ctrue.t() @ (true_delta / (4 * QUAD_SOFT_K) + true_gamma) @ alpha

        idx_linear_B, idx_quad_B = [], []
        for i in range(d.shape[0]):
            if z_true[i] >= 1 / (4 * QUAD_SOFT_K):
                idx_linear_B.append(i)
            elif z_true[i] >= - 1 / (4 * QUAD_SOFT_K):
                idx_quad_B.append(i)
        # print("idx_linear_B:", idx_linear_B)
        # print("idx_quad_B:", idx_quad_B)
        # df_dx = -Ctrue.t() @ delta_true @ diaga @ z_true - Ctrue.t() @ (delta_true / (4 * QUAD_SOFT_K) + gamma_true) @ alpha

        gamma1, gamma2, gamma3 = delta @ diaga @ C @ inver @ beta, inver @ C.t() @ delta @ diaga, inver @ beta
        S1 = torch.cat([inver.unsqueeze(-1) for k in range(C.shape[0])], dim=2) # i * l * k
        S2 = torch.cat([gamma2.unsqueeze(-1) for l in range(C.shape[1])], dim=2) # i * k * l

        for k in range(C.shape[0]):
            S1[:, :, k] *= (eta[k] - gamma1[k, 0])
        for l in range(C.shape[1]):
            S2[:, :, l] *= gamma3[l, 0]
        dx_dC = S1.permute(0, 2, 1) - S2
        """
        A slower version
        dx_dC = torch.zeros(x.shape[0], C.shape[0], C.shape[1])
        for i in range(x.shape[0]): # 20
            for k in range(C.shape[0]):  # 60
                for l in range(C.shape[1]): # 20
                    dx_dC[i, k, l] = -(gamma1[k, 0] * inver[i, l] + gamma3[l, 0] * gamma2[i, k]) + inver[i, l] * eta[k]
                    #              = (eta[k] - gamma1[k, 0]) * inver[i, l] - gamma3[l, 0] * gamma2[i, k]
        """
        for k in range(C.shape[0]):
            df_dC[k, :] = df_dx.t() @ dx_dC[:, k, :].view(dx_dC.shape[0], dx_dC.shape[2])
            # df_dx.t(): 1 * 20
            # dx_dc: 20 * 60 * 20
            # df_dc: (1*) 60 * 20 df_dc[k, :] = 1 * 20 df_dx.t(): 1 * 20 dx_dC[:, k, :] = 20 * 20
        grd = grad_output[0] * df_dC
        return grd[:Ctensor.shape[0], :] - grd[Ctensor.shape[0]:2*Ctensor.shape[0], :], None, None, None, None, None, None, None, None, None

buffer_C, buffer_d, buffer_alpha = None, None, None

def getopt(alpha1, alpha2, A0, b0, C0, d0): # get optimal true value.
    # note: theta is actually alpha1 and alpha0 is actually alpha2.
    global buffer_C, buffer_d, buffer_alpha
    x0 = np.zeros(A0.shape[1])  # 并不是起始点的问题。
    # TODO: optimization and cut duplicated code.
    # C0 is prediction and will change!
    C, d, alpha = merge_constraints(A0, b0, C0, d0, alpha1, alpha2)
    ev = gp.Env(empty=True)
    ev.setParam('OutputFlag', 0)
    ev.start()
    m = gp.Model("matrix1", env=ev)
    # solve twice: first solve the naked problem, then solve it again to get the surrogate optimal.
    x = m.addMVar(shape=C0.shape[1], vtype=GRB.CONTINUOUS, name='x')
    z = m.addMVar(shape=d0.shape[0], vtype=GRB.CONTINUOUS, name='z')
    w = m.addMVar(shape=d0.shape[0], vtype=GRB.CONTINUOUS, name='w')
    m.setObjective(-alpha1.T @ z - alpha2.T @ w, GRB.MAXIMIZE)
    m.addConstr(z >= 0, name="c1")
    m.addConstr(z >= C0 @ x - d0.squeeze(), name='c2')
    m.addConstr(w >= 0, name="c1")
    m.addConstr(w >= d0.squeeze() - C0 @ x, name='c3')
    m.addConstr(x >= 0, name="c3")
    m.addConstr(A0 @ x <= b0.squeeze(), name='c4')
    m.optimize()
    naked_x = x.getAttr('x')
    return getval_twoalpha(naked_x, alpha1, alpha2, A0, b0, C0, d0), naked_x

def resetbuffer():
    global buffer_C, buffer_d, buffer_alpha
    buffer_C, buffer_d, buffer_alpha = None, None, None

def getopt_surro(ground_truth_C, Ctensor, alpha1, alpha2, A0, b0, d0, nograd=False, backlog=None): # the surrogate function
    global buffer_C, buffer_d, buffer_alpha
    C0 = Ctensor.cpu().detach().numpy()
    C, d, alpha_merged = merge_constraints(A0, b0, C0, d0, alpha1, alpha2)
    Ctrue, _, __ = merge_constraints(A0, b0, ground_truth_C, d0, alpha1, alpha2)
    # print("alpha1:",alpha1.shape, "alpha2:",alpha2.shape, "A0:",A0.shape, "b0:", b0.shape, "C0:", C0.shape,"d0:", d0.shape)
    _, x = getopt(alpha1, alpha2, A0, b0, C0, d0)
    # print(_)
    idx_none, idx_quad, idx_linear = [], [], []
    soft_constr = np.matmul(C, x.reshape(-1, 1)) - d
    for i in range(C.shape[0]):
        if -1.0 / (4 * QUAD_SOFT_K) < soft_constr[i] and soft_constr[i] <= 1.0 / (4 * QUAD_SOFT_K):
            idx_quad.append(i)
        elif soft_constr[i] > 1.0 / (4 * QUAD_SOFT_K):
            idx_linear.append(i)
        else: idx_none.append(i)
    # print("idx_quad_A:", idx_quad, "idx_quad_linear:", idx_linear)
    L = d.shape[0]
    delta, gamma = torch.zeros((L, L)), torch.zeros((L, L))
    diagz = torch.zeros((L, L))
    diaga = torch.zeros((L, L))
    for i in idx_quad: delta[i, i] = 2 * QUAD_SOFT_K
    for i in idx_linear: gamma[i, i] = 1
    for i in range(L): diagz[i, i], diaga[i, i] = soft_constr[i, 0], alpha_merged[i, 0]
    try:
        inver = np.linalg.inv(np.matmul(np.matmul(C.T, np.matmul(delta.cpu().numpy(), diaga.cpu().numpy())), C))
    except np.linalg.LinAlgError:
        # print('LinAlgError!', delta, diaga)
        exit(0)

    grd = None if nograd else quad_surro_tensor.apply(Ctensor, Ctrue, inver, diaga, alpha_merged, C, d, delta, gamma)
    # C is the concatenation of Ctensor and others.
    val = getval_twoalpha(x, alpha1, alpha2, A0, b0, C0, d0)

    return val, grd  # grd is the objective value with real theta, while val is the objective value with predicted theta.

def getval(x, alpha0, A0, b0, C0, d0):
    return -alpha0.T.dot(np.maximum(np.matmul(C0, x.reshape(-1, 1)) - d0, 0))

def getval_twoalpha(x, alpha0, alpha1, A0, b0, C0, d0):
    return -(alpha0.T.dot(np.maximum(np.matmul(C0, x.reshape(-1, 1)) - d0, 0)) + alpha1.T.dot(np.maximum(d0 - np.matmul(C0, x.reshape(-1, 1)), 0)))
