import numpy as np
from config import get_K, device, CLIP
import gurobipy
from scipy.optimize import minimize, LinearConstraint
from util import merge_constraints
import torch
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
        QUAD_SOFT_K = get_K()
        theta, truetheta, inver, diaga, alpha, C, d, delta, gamma = args
        inver, diaga, alpha, C, d, delta, gamma = torch.from_numpy(inver).double().to(device), diaga.double().to(device), torch.from_numpy(
            alpha).double().to(device), torch.from_numpy(C).double().to(device), torch.from_numpy(d).double().to(device), delta.double().to(device), gamma.double().to(device)
        # x = (C^T * delta * diaga * C)^(-1) * (theta + C^T * delta * diaga * d - C^T * (1/4k * delta + gamma) * alpha)
        x = inver @ (theta + C.t() @ delta @ diaga @ d - C.t() @ (1.0 / (4 * QUAD_SOFT_K) * delta + gamma) @ alpha)
        z = C @ x - d
        result = truetheta.t() @ x - alpha.t() @ (delta / 2.0 @ ((z + 1.0 / (4.0 * QUAD_SOFT_K)) ** 2) + gamma @ z)
        ctx.save_for_backward(truetheta, x, inver, diaga, alpha, C, d, delta, gamma)
        return result

    @staticmethod
    def backward(ctx, *grad_output):
        QUAD_SOFT_K = get_K()
        torch.set_printoptions(precision=8) # for debugging.
        truetheta, x, inver, diaga, alpha, C, d, delta, gamma = ctx.saved_tensors
        dx_dtheta = inver
        z = C @ x - d
        df_dx = truetheta - C.t() @ delta @ diaga @ z - C.t() @ (delta * 1 / (4 * QUAD_SOFT_K) + gamma) @ alpha
        grd = grad_output[0] * dx_dtheta @ df_dx
        if CLIP == "CLIP" and grd.abs().max() > 0.0001: grd = grd / grd.abs().max() * 0.0001 # normalizing gradients
        elif CLIP == "NORMALIZE" and grd.abs().max() > 0: grd = grd / grd.abs().max() * 0.0001
        return grd, None, None, None, None, None, None, None, None
        # np_dx_dtheta = np.linalg.inv(np.matmul(C.T, np.matmul(delta, np.matmul(diaga, C)))).T
        # np_df_dx = ground_truth_theta.numpy() - np.matmul(C.T, np.matmul(delta, np.matmul(diaga, np.matmul(C, x.reshape(-1, 1)) - d))) - np.matmul(np.matmul(C.T, delta * 1 / (4 * QUAD_SOFT_K) + gamma), alpha)

buffer_C, buffer_d, buffer_alpha = None, None, None

def getopt(theta, alpha0, A0, b0, C0, d0): # get optimal true value.
    # SURROVAL is only not zero at melding, i.e. soft=False, naked=False.
    QUAD_SOFT_K = get_K()
    global buffer_C, buffer_d, buffer_alpha
    x0 = np.zeros(A0.shape[1])  
    if buffer_C is None: buffer_C, buffer_d, buffer_alpha = merge_constraints(A0, b0, C0, d0, alpha0, theta)  # TODO: optimization and cut duplicated code.
    C, d, alpha = buffer_C, buffer_d, buffer_alpha
    ev = gp.Env(empty=True)
    ev.setParam('OutputFlag', 0)
    ev.start()
    m = gp.Model("matrix1", env=ev)
    # solve twice: first solve the naked problem, then solve it again to get the surrogate optimal.
    x = m.addMVar(shape=theta.shape[0], vtype=GRB.CONTINUOUS, name='x')
    z = m.addMVar(shape=d0.shape[0], vtype=GRB.CONTINUOUS, name='z')
    m.setObjective(theta.T @ x - alpha0.T @ z, GRB.MAXIMIZE)
    m.addConstr(z >= 0, name="c1")
    m.addConstr(z >= C0 @ x - d0.squeeze(), name='c2')
    m.addConstr(x >= 0, name="c3")
    m.addConstr(A0 @ x <= b0.squeeze(), name='c4')
    # print(A0, b0, C0, d0, alpha0)
    m.optimize()
    idx_none, idx_quad, idx_linear = [], [], []
    naked_x = x.getAttr('x')
    soft_constr = np.matmul(C, naked_x.reshape(-1, 1)) - d
    for i in range(d.shape[0]):
        # print(i, soft_constr[i])
        if -1.0 / (4 * QUAD_SOFT_K) <= soft_constr[i] and soft_constr[i] <= 1.0 / (4 * QUAD_SOFT_K):
            idx_quad.append(i)
        elif soft_constr[i] > 1.0 / (4 * QUAD_SOFT_K):
            idx_linear.append(i)
        else:
            idx_none.append(i)
    diaga = np.zeros((alpha.shape[0], alpha.shape[0]))
    gamma, delta = np.zeros((C.shape[0], C.shape[0])), np.zeros((C.shape[0], C.shape[0]))
    for i in range(len(idx_linear)):
        gamma[idx_linear[i], idx_linear[i]] = 1
    #print("idx_none soft:", list(filter(lambda x: x < alpha0.shape[0], idx_none)))
    #print("idx_linear soft:", list(filter(lambda x: x < alpha0.shape[0], idx_linear)))
    #print("idx_quad soft:", list(filter(lambda x: x < alpha0.shape[0], idx_quad)))
    for i in range(len(idx_quad)):
        delta[idx_quad[i], idx_quad[i]] = 2 * QUAD_SOFT_K
    for i in range(alpha.shape[0]):
        diaga[i, i] = alpha[i]
    return getval(theta, naked_x, alpha0, A0, b0, C0, d0), naked_x# m.objVal, x.getAttr('x')

def resetbuffer():
    global buffer_C, buffer_d, buffer_alpha
    buffer_C, buffer_d, buffer_alpha = None, None, None

def getopt_surro(ground_truth_theta, thetatensor, alpha0, A0, b0, C0, d0, nograd=False, backlog=None): # the surrogate function
    QUAD_SOFT_K = get_K()
    # print("K:", QUAD_SOFT_K)
    global buffer_C, buffer_d, buffer_alpha
    theta = thetatensor.cpu().detach().numpy()
    if buffer_C is None: buffer_C, buffer_d, buffer_alpha = merge_constraints(A0, b0, C0, d0,
                                                                              alpha0, torch.max(ground_truth_theta, thetatensor).cpu().detach().numpy())  # TODO: optimization and cut duplicated code.
    C, d, alpha = buffer_C, buffer_d, buffer_alpha
    _, x = getopt(theta, alpha0, A0, b0, C0, d0)
    idx_none, idx_quad, idx_linear = [], [], []
    soft_constr = np.matmul(C, x.reshape(-1, 1)) - d
    for i in range(C.shape[0]):
        if -1.0 / (4 * QUAD_SOFT_K) < soft_constr[i] and soft_constr[i] <= 1.0 / (4 * QUAD_SOFT_K):
            idx_quad.append(i)
        elif soft_constr[i] > 1.0 / (4 * QUAD_SOFT_K):
            idx_linear.append(i)
        else: idx_none.append(i)
    L = d.shape[0]
    delta, gamma = torch.zeros((L, L)), torch.zeros((L, L))
    diagz = torch.zeros((L, L))
    diaga = torch.zeros((L, L))
    for i in idx_quad: delta[i, i] = 2 * QUAD_SOFT_K
    for i in idx_linear: gamma[i, i] = 1
    for i in range(L): diagz[i, i], diaga[i, i] = soft_constr[i, 0], alpha[i, 0]

    if(len(idx_quad)) < C.shape[1]: # potential error: the inverse is not full rank!
        if backlog is not None:
            backlog.write("potential singular! " + str(soft_constr) + " " + str(len(idx_quad)) + "\n\n")
            backlog.flush()
            val = getval(ground_truth_theta.cpu().numpy(), x, alpha0, A0, b0, C0, d0)
            return val, None
    ctr = 0
    while True: # if is not full rank
        try:
            inver = np.linalg.inv(np.matmul(np.matmul(C.T, np.matmul(delta.cpu().numpy(), diaga.cpu().numpy())), C)) + ctr * 1e-7
            break
        except np.linalg.LinAlgError:
            ctr += 1
            exit(0)
    grd = None if nograd else quad_surro_tensor.apply(thetatensor, ground_truth_theta, inver, diaga, alpha, C, d, delta, gamma)
    val = getval(ground_truth_theta.cpu().numpy(), x, alpha0, A0, b0, C0, d0)
    return val, grd  # grd is the objective value with real theta, while val is the objective value with predicted theta.

def getval(theta, x, alpha0, A0, b0, C0, d0):
    return theta.T.dot(x) - alpha0.T.dot(np.maximum(np.matmul(C0, x.reshape(-1, 1)) - d0, 0))
