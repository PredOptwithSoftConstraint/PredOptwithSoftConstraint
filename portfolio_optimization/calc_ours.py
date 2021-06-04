import torch
import numpy as np
from config import *
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import math
des=open("results/log.txt","w")
def is_spd(Q):
    return np.all(np.linalg.eigvals(Q) >= 0)

lst_gradnorm, lst_projgradnorm, lst_xmae, lst_Rmae, lst_betamae = [], [], [], [], []

def drawpic():
    global lst_projgradnorm, lst_gradnorm
    plt.title("projgrad L2 norm")
    plt.semilogy([i for i in range(len(lst_projgradnorm))], lst_projgradnorm)
    plt.savefig("results/pic/"+OUTPUT_NAME+"/01.jpg")
    plt.cla()
    plt.title("projgrad L2 norm")
    plt.semilogy([i for i in range(len(lst_gradnorm))], lst_gradnorm)
    plt.savefig("results/pic/"+OUTPUT_NAME+"/02.jpg")
    plt.cla()

class quad_surro_tensor(torch.autograd.Function): # with gradient.
    @staticmethod
    def forward(ctx, *args):
        global QUAD_SOFT_K
        A, b, C, d, alpha,  C0, d0, alpha0, Q, Q_real, theta, theta_real, KK = args
        if KK is not None:
            QUAD_SOFT_K = KK
        #if not is_spd(Q):
        #    raise ValueError("Das ist nicht ein SPD matrix!")
        #if not is_spd(Q_real):
        #    raise ValueError("Das ist nicht ein SPD matrix 2!")
        theta, theta_real = theta.view(-1, 1), theta_real.view(-1, 1)
        x = getopt(A.numpy(), b.numpy(), Q.detach().numpy(), theta.detach().numpy(), softcon=(C0, d0, alpha0))
        old_constr = C @ torch.from_numpy(x).double() - d
        idx_none, idx_quad, idx_linear = [], [], []
        for i in range(C.shape[0]):
            if old_constr[i] >= - 1 / (4 * QUAD_SOFT_K) and old_constr[i] <= 1 / (4 * QUAD_SOFT_K): idx_quad.append(i)
            elif old_constr[i] >= 1 / (4 * QUAD_SOFT_K): idx_linear.append(i)
            else: idx_none.append(i)
        M = C.shape[0]
        diaga, delta, gamma = torch.zeros(M, M).double(), torch.zeros(M, M).double(), torch.zeros(M, M).double()
        for i in idx_linear:
            gamma[i, i] = 1
        for i in idx_quad:
            delta[i, i] = 2 * QUAD_SOFT_K
        #if len(idx_linear) > 0:
        #    raise ValueError("Error!")
        print("idx_linear:",idx_linear)
        for i in range(M):
            diaga[i, i] = alpha[i, 0]
        # x = (C^T * delta * diaga * C)^(-1) * (theta + C^T * delta * diaga * d - C^T * (1/4k * delta + gamma) * alpha)
        Rinv, Rinv_real = 2 * Q + C.t() @ delta @ diaga @ C, 2 * Q_real + C.t() @ delta @ diaga @ C
        R, R_real = torch.inverse(Rinv), torch.inverse(Rinv_real)
        beta = (theta + C.t() @ delta @ diaga @ d - C.t() @ (gamma + delta / (4 * QUAD_SOFT_K)) @ alpha)
        beta_real = (theta_real + C.t() @ delta @ diaga @ d - C.t() @ (gamma + delta / (4 * QUAD_SOFT_K)) @ alpha)
        new_x = R @ beta
        new_constr = C @ new_x - d
        result = theta_real.t() @ new_x - new_x.t() @ Q @ new_x - alpha.t() @ new_constr
        ctx.save_for_backward(theta_real, Q.detach(), Q_real, new_x, R, Rinv, R_real, Rinv_real, beta, beta_real, alpha, C, d, diaga, delta, gamma, new_constr)
        return result

    @staticmethod
    def backward(ctx, *grad_output):
        global des, lst_gradnorm, lst_projgradnorm, lst_xmae, lst_Rmae, lst_betamae
        torch.set_printoptions(precision=8) # for debugging.
        theta_real, Q, Q_real, new_x, R, Rinv, R_real, Rinv_real, beta, beta_real, alpha, C, d, diaga, delta, gamma, new_constr = ctx.saved_tensors
        df_dx = beta_real - Rinv_real @ new_x
        dx_dtheta = R
        df_dtheta = (grad_output[0] * dx_dtheta @ df_dx).squeeze()
        p = R.t() @ (Rinv_real @ new_x - beta_real)

        df_dq = grad_output[0] * 2 * p @ new_x.t()
        f=open("why_ours","a")
        f.write("DFDQ:"+str(df_dq.detach().numpy())+"\n\n")
        f.close()
        """
        v0 = torch.norm(df_dq, 2)
        proj_df_dq = (df_dq + df_dq.t()) / 2
        steplen = 1e-5
        diff_Q = Q + steplen * df_dq
        diff_R = torch.inverse(2 * diff_Q + C.t() @ delta @ diaga @ C)  # the should-be value of R. opt_x = diff_R * diff_beta = diff_proj_R * calibrated beta.
        diff_beta = beta + steplen * df_dtheta.view(-1, 1)              # the actual update of beta.
        diff_proj_Q = Q + steplen * proj_df_dq
        diff_proj_R = torch.inverse(2 * diff_proj_Q + C.t() @ delta @ diaga @ C)
        calibrated_beta = torch.inverse(diff_proj_R) @ diff_R @ diff_beta # the calibrated, "real" delta.
        df_dtheta = ((calibrated_beta - beta) / steplen).view(-1)  # correction to the projection
        # print(diff_beta)
        lst_gradnorm.append(v0)
        lst_projgradnorm.append(torch.norm(proj_df_dq, 2))
        diff1, diff2, diff3 = torch.nn.MSELoss()(beta, beta_real), torch.nn.MSELoss()(R, R_real), torch.nn.MSELoss()(R @ beta, R_real @ beta_real)
        lst_betamae.append(diff1)
        lst_Rmae.append(diff2)
        lst_xmae.append(diff3)
        
        #if diff3 < 1e-4:
        #    print("diff1:",diff1, "diff2:", diff2, "diff3:", diff3)
        if v0 < 1e-8:
            new_optimal_x = R_real @ beta_real
            v1 = getval(Q_real, theta_real, new_optimal_x)
            v2 = getval(Q_real, theta_real, new_x)
            raise ValueError("Error!")
            exit(0)
        """
        #for i in range(df_dq.shape[0]):
        #    for j in range(i+1,df_dq.shape[1]):
        #        df_dq[i, j] = df_dq[j, i]
        # df_dq = (df_dq + df_dq.t()) / 2
        #print("svd of df_dq:", torch.svd(df_dq), "\n")
        #print("df_dq:" ,df_dq,"\n")
        #exit(0)
        return None, None, None,None, None, None, None, None, df_dq, None, df_dtheta, None, None
        # np_dx_dtheta = np.linalg.inv(np.matmul(C.T, np.matmul(delta, np.matmul(diaga, C)))).T
        # np_df_dx = ground_truth_theta.numpy() - np.matmul(C.T, np.matmul(delta, np.matmul(diaga, np.matmul(C, x.reshape(-1, 1)) - d))) - np.matmul(np.matmul(C.T, delta * 1 / (4 * QUAD_SOFT_K) + gamma), alpha)

def getval(Q, theta, x, softcon):
    v = theta.t() @ x - x.t() @ Q @ x
    if softcon is not None:
        C0, d0, alpha0 = softcon
        C0, d0, alpha0 = C0.double(), d0.double(), alpha0.double()
        v -= alpha0.t() @ torch.maximum(C0 @ x - d0, torch.zeros_like(d0).double())
        # print((C0 @ x - d0).shape)
    return v
"""
def getopt_surrogate(A, b, Q, theta):
    n = theta.shape[0]
    C, d, alpha = A, b, (0.2 * math.sqrt(n) * torch.ones(n + 2, 1)).double()
    old_x = getopt(A.numpy(), b.numpy(), Q.detach().numpy(), theta.detach().numpy())
    old_constr = C @ torch.from_numpy(old_x).double() - d
    idx_none, idx_quad, idx_linear = [], [], []
    for i in range(C.shape[0]):
        if old_constr[i] >= - 1 / (4 * QUAD_SOFT_K) and old_constr[i] <= 1 / (4 * QUAD_SOFT_K):
            idx_quad.append(i)
        elif old_constr[i] >= 1 / (4 * QUAD_SOFT_K):
            idx_linear.append(i)
        else:
            idx_none.append(i)
    M = C.shape[0]
    diaga, delta, gamma = torch.zeros(M, M).double(), torch.zeros(M, M).double(), torch.zeros(M, M).double()
    for i in idx_linear:
        gamma[i, i] = 1
    for i in idx_quad:
        delta[i, i] = 2 * QUAD_SOFT_K
    if len(idx_linear) > 0:
        raise ValueError("Error!")
    for i in range(M):
        diaga[i, i] = alpha[i, 0]
    # x = (C^T * delta * diaga * C)^(-1) * (theta + C^T * delta * diaga * d - C^T * (1/4k * delta + gamma) * alpha)
    Rinv = 2 * Q + C.t() @ delta @ diaga @ C
    R = torch.inverse(Rinv)
    beta = (theta.view(-1, 1) + C.t() @ delta @ diaga @ d - C.t() @ (gamma + delta / (4 * QUAD_SOFT_K)) @ alpha)
    return R @ beta
"""
def getopt(A, b, Q, theta, softcon): # get optimal true value.
    # SURROVAL is only not zero at melding, i.e. soft=False, naked=False.
    ev = gp.Env(empty=True)
    ev.setParam('OutputFlag', 0)
    ev.start()
    m = gp.Model("matrix1", env=ev)
    if softcon is None:
        # solve twice: first solve the naked problem, then solve it again to get the surrogate optimal.
        x = m.addMVar(shape=theta.shape[0], vtype=GRB.CONTINUOUS, name='x')
        m.setObjective(theta.T @ x - x @ Q @ x, GRB.MAXIMIZE)
        m.addConstr(A @ x <= b.squeeze(), name='c4')
        m.optimize()
        naked_x = x.getAttr('x')
    else:
        # solve twice: first solve the naked problem, then solve it again to get the surrogate optimal.
        C, d, alpha = softcon
        # print(C.shape, d.shape, alpha.shape)
        C, d = C.numpy(), d.numpy()
        alpha = alpha.numpy()
        x = m.addMVar(shape=theta.shape[0], vtype=GRB.CONTINUOUS, name='x')
        z = m.addMVar(shape=d.shape[0], vtype=GRB.CONTINUOUS, name='z')
        m.setObjective(theta.T @ x - x @ Q @ x - alpha.T @ z, GRB.MAXIMIZE)
        m.addConstr(A @ x <= b.squeeze(), name='c4')
        m.addConstr(z >= 0)
        m.addConstr(z >= C @ x - d.squeeze())
        m.optimize()
        naked_x = x.getAttr('x')
    return naked_x.reshape(-1, 1)
