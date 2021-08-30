from calc import getopt, getval
import time
import torch
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from config import device
total_time, total_invoke = 0, 0

def solve(theta, theta_true, alpha, A, b, C, d):
    # current: cvxpy v1.1.6
    m = A.shape[0]
    m2 = C.shape[0]
    n = C.shape[1]
    x_var = cp.Variable(n)
    theta_para = cp.Parameter(n)
    constraints = [A @ x_var <= b.reshape(-1), x_var >= 0]
    # print(alpha)
    objective = cp.Maximize(theta_para @ x_var - alpha.T @ cp.maximum(C @ x_var - d.reshape(-1), 0) - 0.1 * cp.sum_squares(x_var))
    problem = cp.Problem(objective, constraints)
    cvxpylayer = CvxpyLayer(problem, parameters=[theta_para], variables=[x_var])
    x, = cvxpylayer(theta)
    obj = theta_true @ x - torch.from_numpy(alpha.T).to(device) @ torch.maximum(torch.from_numpy(C).to(device) @ x.view(-1, 1) - torch.from_numpy(d).to(device), torch.zeros(m2, 1).to(device))
    return obj

def getopt_DF(ground_truth_theta, thetatensor, alpha0, A0, b0, C0, d0):

    global total_time, total_invoke
    t0 = time.time()
    theta = thetatensor.cpu().detach().numpy()
    _, x = getopt(theta, alpha0, A0, b0, C0, d0)
    grd = solve(thetatensor.to(device), torch.from_numpy(ground_truth_theta).to(device), alpha0, A0, b0, C0, d0)
    val = getval(ground_truth_theta, x, alpha0, A0, b0, C0, d0)
    total_time, total_invoke = total_time + time.time() - t0, total_invoke + 1
    #print("getopt_DF average time:", total_time / total_invoke)
    gt = getopt(ground_truth_theta, alpha0, A0, b0, C0, d0)
    #print("val:", val, "groundtruth_val:", gt[0])
    #print("x:", x, "groundtruth_x:", gt[1])
    return val, grd
