import torch
import tqdm
import time
from utils import computeCovariance
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import sys
import pandas as pd
import torch
import numpy as np
import qpth
import scipy
import cvxpy as cp
import random
import argparse
import tqdm
import time
import math
import datetime as dt
from cvxpylayers.torch import CvxpyLayer
import matplotlib.pyplot as plt
import torch.nn
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import gurobipy as gp
from gurobipy import GRB
from utils import normalize_matrix, normalize_matrix_positive, normalize_vector, normalize_matrix_qr, normalize_projection
from sqrtm import sqrtm
from calc_ours import quad_surro_tensor, getval, getopt, drawpic
from config import RATIO, OUTPUT_NAME
alp = 2
REG = 0.1
solver = 'cvxpy'

MAX_NORM = 0.1
T_MAX_NORM = 0.1

def symsqrt(matrix):
    """Compute the square root of a positive definite matrix."""
    _, s, v = matrix.svd()
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

def computeCovariance(covariance_mat):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    n = len(covariance_mat)
    cosine_matrix = torch.zeros((n,n))
    for i in range(n):
        cosine_matrix[i] = cos(covariance_mat, covariance_mat[i].repeat(n,1))
    return cosine_matrix

def generateDataset(data_loader, n=200, num_samples=100):
    feature_mat, target_mat, feature_cols, covariance_mat, target_name, dates, symbols = data_loader.load_pytorch_data()
    symbol_indices = np.random.choice(len(symbols), n, replace=False)
    feature_mat    = feature_mat[:num_samples,symbol_indices]
    target_mat     = target_mat[:num_samples,symbol_indices]
    covariance_mat = covariance_mat[:num_samples,symbol_indices]
    symbols = [symbols[i] for i in symbol_indices]
    dates = dates[:num_samples]

    num_samples = len(dates)

    sample_shape, feature_size = feature_mat.shape, feature_mat.shape[-1]

    # ------ normalization ------
    feature_mat = feature_mat.reshape(-1,feature_size)
    feature_mat = (feature_mat - torch.mean(feature_mat, dim=0)) / (torch.std(feature_mat, dim=0) + 1e-5) 
    feature_mat = feature_mat.reshape(sample_shape, feature_size)

    dataset = data_utils.TensorDataset(feature_mat, covariance_mat, target_mat)

    indices = list(range(num_samples))
    # np.random.shuffle(indices)

    train_size, validate_size = int(num_samples * 0.7), int(num_samples * 0.1)
    train_indices    = indices[:train_size]
    validate_indices = indices[train_size:train_size+validate_size]
    test_indices     = indices[train_size+validate_size:]

    batch_size = 1
    train_dataset    = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    validate_dataset = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validate_indices))
    test_dataset     = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))

    # train_dataset    = dataset[train_indices]
    # validate_dataset = dataset[validate_indices]
    # test_dataset     = dataset[test_indices]

    return train_dataset, validate_dataset, test_dataset

rec_features, rec_covmat, rec_labels = None, None, None # for training set to repeat the same datapoint.
plts, mseQ, maeQ, msetheta, maetheta, grdnm, opt_mae = [], [], [], [], [], [], []
turn = 0

def ours_train_portfolio(model, covariance_model, optimizer, epoch, dataset, device='cpu', evaluate=False, KK=None, AP=None, softcon=None):
    global plts, mseQ, maeQ, msetheta, maetheta, grdnm, opt_mae
    if softcon is not None:
        C0, d0, alpha0 = softcon
        C0, d0, alpha0 = C0.double(), d0.double(), alpha0.double()
    opts = []
    model.train()
    covariance_model.train()
    loss_fn = torch.nn.MSELoss() # torch.nn.MSELoss()
    train_losses, train_objs = [], []
    global turn, rec_features, rec_covmat, rec_labels
    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    repeat, lianxu = False, True # for debugging. repeat - training only with the first data point; lianxu - training Q and theta simutaneously (False is separately in turn).
    if AP is None: AP = 0.2 # a constant for the generation of alpha. This coefficient is much smaller and practical than theoretical upper bound, yet still working.
    with tqdm.tqdm(dataset) as tqdm_loader:

        batchloss = torch.zeros(1)
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            if rec_features is None:
                rec_features, rec_covmat, rec_labels = features, covariance_mat, labels
            if repeat:
                features, covariance_mat, labels = rec_features, rec_covmat, rec_labels
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).double() # only one single data

            n = len(covariance_mat)
            Q_real = (1/RATIO) * alp/2 * (computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n).double() * REG)
            predictions = model(features.double())[:,0]
            loss = loss_fn(predictions, labels)
            Q = (1/RATIO) * alp/2 * (covariance_model() * (1 - REG) + torch.eye(n).double() * REG)  # TODO
            A, b = torch.cat((-torch.ones(1, n), torch.ones(1, n), -torch.eye(n)), dim=0).double(), torch.cat(
                (-RATIO*torch.ones(1, 1), RATIO*torch.ones(1, 1), torch.zeros(n, 1)), dim=0).double()
            if softcon is None:
                C, d, alpha = A, b, (AP * math.sqrt(n) * torch.ones(n + 2, 1)).double()
            else:
                C, d, alpha = torch.cat((A, C0), dim=0).double(), torch.cat((b, d0), dim=0).double(), torch.cat(((AP * math.sqrt(n) * torch.ones(n + 2, 1).double()), alpha0), dim=0).double()
                softcon = C0, d0, alpha0
            if repeat: # control the variant
                predictions = labels
                pass
            elif not lianxu:
                if batch_idx % 2 == 0: # train the two components in turn.
                    predictions = labels
                else:
                    Q = Q_real
            x, opt_x = getopt(A.numpy(), b.numpy(), Q.detach().numpy(), predictions.view(-1, 1).detach().numpy(), softcon), getopt(A.numpy(), b.numpy(), Q_real.numpy(), labels.view(-1, 1).numpy(), softcon)
            val = getval(Q_real, labels.view(-1, 1), torch.from_numpy(x).double(), softcon)
            if val > getval(Q_real, labels.view(-1, 1), torch.from_numpy(opt_x).double(), softcon):
                exit(0)
            optval = getval(Q_real, labels.view(-1, 1), torch.from_numpy(opt_x).double(), softcon)
            opt_mae.append(torch.nn.L1Loss()(torch.from_numpy(x).double(), torch.from_numpy(opt_x).double()).item())
            plts.append((optval - val)/RATIO)
            if evaluate:
                forward_time += time.time() - forward_start_time
                inference_start_time = time.time()
                # FIXME:solve with gurobi and finally set obj = xxx.
                if softcon is None:
                    C0, d0, alpha0 = 0, 0, 0
                obj = quad_surro_tensor.apply(A, b, C, d, alpha, C0, d0, alpha0, Q, Q_real, predictions, labels, KK)
                val = getval(Q_real, labels.view(-1, 1), torch.from_numpy(getopt(A.numpy(), b.numpy(), Q.detach().numpy(), predictions.view(-1, 1).detach().numpy(), softcon)).double(), softcon)
                inference_time += time.time() - inference_start_time
            else:
                obj = torch.Tensor([0])

            # ====================== back-prop =====================
            optimizer.zero_grad()
            backward_start_time = time.time()
            BATCH_SIZE = 1 if not repeat else 1
            reg = 0
            for parameter in model.parameters():
                reg += torch.norm(parameter, 1)
            for parameter in covariance_model.parameters():
                reg += torch.norm(parameter, 1)
            batchloss = batchloss - obj #+ 0.0001 * reg # + 1 * torch.nn.L1Loss()(Q, Q_real) + 10 * torch.nn.L1Loss()(predictions, labels) # now with regularization term.
            train_objs.append(val.item() / RATIO)
            opts.append(optval / RATIO)
            if batch_idx % BATCH_SIZE == BATCH_SIZE - 1 or batch_idx % 2028 == 2027:
                batchloss.backward()
                gradnorm = 0
                if repeat:
                    #for parameter in model.parameters():
                    #    parameter.grad = torch.clamp(parameter.grad, min=-0.2 * MAX_NORM, max=0.2 * MAX_NORM)
                    msetheta.append(torch.nn.MSELoss()(predictions, labels).item())
                    maetheta.append(torch.nn.L1Loss()(predictions, labels).item())

                    for parameter in covariance_model.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-0.2*MAX_NORM, max=0.2*MAX_NORM)
                        gradnorm += torch.norm(parameter.grad, 2)
                    grdnm.append(gradnorm.item())
                    mseQ.append(torch.nn.MSELoss()((RATIO ** 2) * Q,(RATIO ** 2) * Q_real).item())
                    maeQ.append(torch.nn.L1Loss()((RATIO ** 2) * Q, (RATIO ** 2) * Q_real).item())
                else:
                    if not lianxu:
                        if batch_idx % 2 == 1:
                            for parameter in model.parameters():
                                parameter.grad = torch.clamp(parameter.grad, min=-0.2*MAX_NORM, max=0.2*MAX_NORM)
                            msetheta.append(torch.nn.MSELoss()(RATIO * predictions, RATIO * labels).item())
                            maetheta.append(torch.nn.L1Loss()(RATIO * predictions, RATIO * labels).item())
                        else:
                            for parameter in covariance_model.parameters():
                                parameter.grad = torch.clamp(parameter.grad, min=-0.1*MAX_NORM, max=0.1*MAX_NORM)
                            mseQ.append(torch.nn.MSELoss()((RATIO ** 2) * Q, (RATIO ** 2) * Q_real).item())
                            maeQ.append(torch.nn.L1Loss()((RATIO ** 2) * Q, (RATIO ** 2) * Q_real).item())
                    else:
                        for parameter in model.parameters():
                            parameter.grad = torch.clamp(parameter.grad, min=-0.2 * MAX_NORM, max=0.2 * MAX_NORM)
                            gradnorm += torch.norm(parameter.grad, 2)
                        for parameter in covariance_model.parameters():
                            parameter.grad = torch.clamp(parameter.grad, min=-0.2 * MAX_NORM, max=0.2 * MAX_NORM)
                            gradnorm += torch.norm(parameter.grad, 2)
                        grdnm.append(gradnorm.item())
                        msetheta.append(torch.nn.MSELoss()(predictions, labels).item())
                        maetheta.append(torch.nn.L1Loss()(predictions, labels).item())
                        mseQ.append(torch.nn.MSELoss()(Q, Q_real).item())
                        maeQ.append(torch.nn.L1Loss()(Q, Q_real).item())
                optimizer.step()
                backward_time += time.time() - backward_start_time
                train_losses.append(loss.item())

                tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{val.item()/RATIO*100:.6f}%',optimal=f'{optval.item()/RATIO*100:.6f}%') # this is val! not obj!
            batchloss = torch.zeros(1)
            if batch_idx % 2028 == 2027: # for debugging.
                print("maeQ:", sum(maeQ[-2028:])/2028, "maetheta:", sum(maetheta[-2028:])/2028)
                drawpic()
                print("turn: #",turn)
                plt.title("MAE of matrix Q")
                plt.plot([i for i in range(len(maeQ))], maeQ)
                name="onetarget"
                plt.savefig("results/pic/"+OUTPUT_NAME+"/"+name+"_maeQ_"+str(turn)+".pdf")
                plt.cla()
                plt.title("The total L2 norm of gradients")
                plt.semilogy([i for i in range(len(grdnm))], grdnm)
                plt.savefig("results/pic/"+OUTPUT_NAME+"/"+name+"_gradnorm_"+str(turn)+".pdf")
                plt.cla()
                plt.title("MAE of theta")
                plt.plot([i for i in range(len(maetheta))], maetheta)
                plt.savefig("results/pic/"+OUTPUT_NAME+"/"+name+"_maetheta_"+str(turn)+".pdf")
                plt.cla()
                plt.title("MAE of optimal x (sum(x)=20)")
                plt.semilogy([i for i in range(len(opt_mae))], opt_mae)
                plt.savefig("results/pic/"+OUTPUT_NAME+"/" + name + "_optMAE_" + str(turn) + ".pdf")
                plt.cla()
                plt.title("MSE of Q")
                plt.plot([i for i in range(len(maeQ))], mseQ)
                plt.savefig("results/pic/"+OUTPUT_NAME+"/"+name+"_mseQ_" + str(turn) + ".pdf")
                plt.cla()
                plt.title("MSE of theta")
                plt.plot([i for i in range(len(maetheta))], msetheta)
                plt.savefig("results/pic/"+OUTPUT_NAME+"/"+name+"_msetheta_" + str(turn) + ".pdf")
                plt.cla()
                plt.title("regret")
                plt.semilogy([i for i in range(len(plts))], plts)
                plt.savefig("results/pic/"+OUTPUT_NAME+"/" + name + "_regret_" + str(turn) + ".pdf")
                plt.cla()
                print("total regret:", sum(plts).item(), "avg per case:", sum(plts).item() / 2028 * 100,"%")
                # plts, maeQ, mseQ, maetheta, msetheta = [], [], [], [], []
        turn += 1
    average_loss    = np.mean(train_losses)
    average_obj     = np.mean(train_objs)
    print("train_objs:", train_objs, "opts:", np.mean(opts), len(train_objs), len(opts))
    return average_loss, average_obj, (forward_time, inference_time, qp_time, backward_time)

tsdf_Q_losses, tsdf_theta_losses = [], []

def train_portfolio(model, covariance_model, optimizer, epoch, dataset, training_method='two-stage', device='cpu', evaluate=False, softcon=None):
    if softcon is not None:
        C0, d0, alpha0 = softcon
        C0, d0, alpha0 = C0.double(), d0.double(), alpha0.double()
    model.train()
    covariance_model.train()
    loss_fn = torch.nn.L1Loss() if training_method == 'two-stage'else torch.nn.MSELoss()  # not MSE!
    train_losses, train_objs = [], []
    global tsdf_Q_losses, tsdf_theta_losses
    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    Q_losses, theta_losses = 0, 0
    global rec_features, rec_covmat, rec_labels
    repeat = False
    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).double() # only one single data
            if rec_features is None:
                rec_features, rec_covmat, rec_labels = features, covariance_mat, labels
            if repeat:
                features, covariance_mat, labels = rec_features, rec_covmat, rec_labels
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n).double() * REG
            predictions = model(features.double())[:,0]
            loss = loss_fn(predictions, labels)
            Q = covariance_model() * (1 - REG) + torch.eye(n).double() * REG  # TODO

            if repeat: # control the variant
                predictions = labels
                pass

            if evaluate:
                forward_time += time.time() - forward_start_time
                inference_start_time = time.time()

                A, b = torch.cat((-torch.ones(1, n), torch.ones(1, n), -torch.eye(n)), dim=0).double(), torch.cat(
                    (-torch.ones(1, 1), torch.ones(1, 1), torch.zeros(n, 1)), dim=0).double()
                x_opt = torch.from_numpy(getopt(A.numpy(), b.numpy(), Q.detach().numpy(), predictions.view(-1, 1).detach().numpy(), softcon)).double().view(-1, 1)
                obj_nake = getval(Q, predictions, x_opt, softcon)
                idx_linear, gamma = [], torch.zeros(d0.shape[0], d0.shape[0]).double()
                old_constr = C0 @ x_opt - d0
                for i in range(d0.shape[0]):
                    if old_constr[i] > 0:
                        idx_linear.append(i)
                        gamma[i, i] = 1
                if training_method == "decision-focused":
                    # to build the function. It is actually the limit situation where the smooth K is infinitely large.
                    C = gamma @ C0
                    x_var = cp.Variable(n)
                    L_para = cp.Parameter((n, n))
                    p_para = cp.Parameter(n)
                    p = predictions
                    L = sqrtm(Q)
                    constraints = [x_var >= 0, x_var <= 1, cp.sum(x_var) == 1]
                    v1, v2 = 0.5 * alp, alpha0.T @ C
                    objective = cp.Minimize(v1 * cp.sum_squares(L_para @ x_var) + p_para.T @ x_var + v2 @ x_var)
                    problem = cp.Problem(objective, constraints)
                    cvxpylayer = CvxpyLayer(problem, parameters=[L_para, p_para], variables=[x_var])
                    x, = cvxpylayer(L, -p)

                else:
                    x = x_opt

                obj = labels @ x - 0.5 * alp * x.t() @ Q_real @ x - alpha0.t() @  torch.maximum(C0 @ x.view(-1, 1) - d0, torch.zeros_like(d0))
                print(obj)

                inference_time += time.time() - inference_start_time
            else:
                obj = torch.Tensor([0])
            """
            d = open("why_DF.txt", "a")
            if batch_idx == 0:
                d.write(str(Q_real.numpy()) + "\n\n")
                # d.write(str(labels.numpy()) + "\n\n")
            d.write(str(Q.detach().numpy()) + "\n\n")
            """
            # ====================== back-prop =====================
            optimizer.zero_grad()
            backward_start_time = time.time()
            try:
                Q_loss = torch.norm(Q - Q_real)
                Q_losses += Q_loss.item()
                theta_losses += loss.item()
                if training_method in ['two-stage', 'two-stage2']:
                    (loss + Q_loss).backward()
                elif training_method == 'decision-focused':
                    (-obj).backward()
                    # (-obj + loss).backward() # TODO
                    for parameter in model.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-MAX_NORM, max=MAX_NORM)
                    for parameter in covariance_model.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-MAX_NORM, max=MAX_NORM)
                else:
                    raise ValueError('Not implemented method')
            except:
                print("no grad is backpropagated...")
                pass
            optimizer.step()
            backward_time += time.time() - backward_start_time

            train_losses.append(loss.item())
            train_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj_nake.item()*100:.6f}%')
    tsdf_theta_losses.append(theta_losses/2028)
    tsdf_Q_losses.append(Q_losses/2028)
    print("theta losses:", tsdf_theta_losses[-1])
    print("Q losses:", tsdf_Q_losses[-1])
    average_loss    = np.mean(train_losses)
    average_obj     = np.mean(train_objs)
    return average_loss, average_obj, (forward_time, inference_time, qp_time, backward_time)

def ours_validate_portfolio(model, covariance_model, scheduler, epoch, dataset, device='cpu',
                       evaluate=False, softcon=None):
    model.eval()
    covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    validate_losses, validate_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0, :, 0].to(
                device).double()  # only one single data
            n = len(covariance_mat)
            Q_real = alp/2 * (computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n).double() * REG)
            predictions = model(features.double())[:, 0]

            loss = loss_fn(predictions, labels)

            if evaluate:
                Q = covariance_model() * 2 * (1 - REG) + torch.eye(n).double() * 2 * REG
                inference_start_time = time.time()
                # FIXME
                A, b = torch.cat((-torch.ones(1, n), torch.ones(1, n), -torch.eye(n)), dim=0).double(), torch.cat(
                    (-torch.ones(1, 1),torch.ones(1, 1), torch.zeros(n, 1)), dim=0).double()
                obj = getval(Q_real, labels.view(-1, 1), torch.from_numpy(getopt(A.numpy(), b.numpy(), Q.detach().numpy(), predictions.view(-1, 1).detach().numpy(), softcon)).double(), softcon)
                inference_time += time.time() - inference_start_time
            else:
                obj = torch.Tensor([0])

            validate_losses.append(loss.item())
            validate_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item() * 100:.6f}%')

    average_loss = np.mean(validate_losses)
    average_obj = np.mean(validate_objs)

    if (epoch > 0):
        scheduler.step(-average_obj)

    return average_loss, average_obj  # , (forward_time, inference_time, qp_time, backward_time)


def validate_portfolio(model, covariance_model, scheduler, epoch, dataset, training_method='two-stage', device='cpu', evaluate=False, softcon=None):
    model.eval()
    covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    validate_losses, validate_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).double() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n).double() * REG
            predictions = model(features.double())[:,0]

            loss = loss_fn(predictions, labels)

            if evaluate:
                Q = covariance_model() * (1 - REG) + torch.eye(n).double() * REG

                forward_time += time.time() - forward_start_time
                inference_start_time = time.time()
                A, b = torch.cat((-torch.ones(1, n), torch.ones(1, n), -torch.eye(n)), dim=0).double(), torch.cat(
                    (-torch.ones(1, 1), torch.ones(1, 1), torch.zeros(n, 1)), dim=0).double()

                obj = getval(Q_real, labels.view(-1, 1), torch.from_numpy(
                    getopt(A.numpy(), b.numpy(), Q.detach().numpy(), predictions.view(-1, 1).detach().numpy(),
                           softcon)).double(), softcon)

                inference_time += time.time() - inference_start_time
            else:
                obj = torch.Tensor([0])

            validate_losses.append(loss.item())
            validate_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item()*100:.6f}%')

    average_loss    = np.mean(validate_losses)
    average_obj     = np.mean(validate_objs)
   
    if (epoch > 0):
        if training_method == "two-stage":
            scheduler.step(average_loss)
        elif training_method == "decision-focused" or training_method == "surrogate":
            scheduler.step(-average_obj)
        else:
            raise TypeError("Not Implemented Method")
   
    return average_loss, average_obj # , (forward_time, inference_time, qp_time, backward_time)

def ours_test_portfolio(model, covariance_model, epoch, dataset, device='cpu', evaluate=False, softcon=None):
    model.eval()
    covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []
    test_opts = []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    total_regret = 0
    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device).double(), covariance_mat[0].to(device).double(), labels[0,:,0].to(device).double() # only one single data
            n = len(covariance_mat)
            Q_real = alp/2 * (computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n).double() * REG)

            if epoch == -1:
                predictions = labels
                Q = Q_real
            else:
                predictions = model(features.double())[:,0]
                Q = covariance_model() * 2 * (1 - REG) + torch.eye(n).double() * 2 * REG

            loss = loss_fn(predictions, labels)

            if evaluate:
                forward_time += time.time() - forward_start_time
                inference_start_time = time.time()

                #FIXME
                A, b = torch.cat((-torch.ones(1, n), torch.ones(1, n), -torch.eye(n)), dim=0).double(), torch.cat(
                    (-torch.ones(1, 1), torch.ones(1, 1), torch.zeros(n, 1)), dim=0).double()
                obj = getval(Q_real, labels.view(-1, 1), torch.from_numpy(getopt(A.numpy(), b.numpy(), Q.detach().numpy(), predictions.view(-1, 1).detach().numpy(), softcon)).double(), softcon)
                optobj = getval(Q_real, labels.view(-1, 1), torch.from_numpy(getopt(A.numpy(), b.numpy(), Q_real.detach().numpy(), labels.view(-1, 1).detach().numpy(), softcon)).double(), softcon)
                total_regret += optobj - obj
                inference_time += time.time() - inference_start_time
            else:
                obj = torch.Tensor([0])

            test_losses.append(loss.item())
            test_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item()*100:.6f}%')
    print("test total regret:", total_regret, total_regret / 581)
    # print('opts:', test_opts)
    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)
    return average_loss, average_obj # , (forward_time, inference_time, qp_time, backward_time)

def test_portfolio(model, covariance_model, epoch, dataset, device='cpu', evaluate=False, softcon=None):
    model.eval()
    covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []
    test_opts = []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).double() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n).double() * REG

            if epoch == -1:
                predictions = labels
                Q = Q_real
            else:
                predictions = model(features.double())[:,0]
                Q = covariance_model() * (1 - REG) + torch.eye(n).double() * REG

            loss = loss_fn(predictions, labels)

            if evaluate:
                forward_time += time.time() - forward_start_time
                inference_start_time = time.time()
                A, b = torch.cat((-torch.ones(1, n), torch.ones(1, n), -torch.eye(n)), dim=0).double(), torch.cat(
                    (-torch.ones(1, 1), torch.ones(1, 1), torch.zeros(n, 1)), dim=0).double()

                obj = getval(Q_real, labels.view(-1, 1), torch.from_numpy(
                    getopt(A.numpy(), b.numpy(), Q.detach().numpy(), predictions.view(-1, 1).detach().numpy(),
                           softcon)).double(), softcon)
                inference_time += time.time() - inference_start_time
            else:
                obj = torch.Tensor([0])

            test_losses.append(loss.item())
            test_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item()*100:.6f}%') 

    # print('opts:', test_opts)
    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)
    return average_loss, average_obj # , (forward_time, inference_time, qp_time, backward_time)

