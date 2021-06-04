# Taken from Wilder et al.'s work https://github.com/guaguakai/surrogate-optimization-learning/blob/master/portfolio/model.py
import os
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

from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_utils import SP500DataLoader
from portfolio_utils import computeCovariance, generateDataset
from model import PortfolioModel, CovarianceModel
from portfolio_utils import train_portfolio, validate_portfolio, test_portfolio, ours_train_portfolio, ours_validate_portfolio, ours_test_portfolio
from utils import normalize_matrix, normalize_matrix_positive, normalize_vector, normalize_matrix_qr, normalize_projection
from config import OUTPUT_NAME

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

if __name__ == '__main__':
    # ==================== Parser setting ==========================
    parser = argparse.ArgumentParser(description='Portfolio Optimization')
    parser.add_argument('--filepath', type=str, default='', help='filename under folder results')
    parser.add_argument('--method', type=int, default=3, help='0: two-stage(L1), 1: decision-focused,2=two-stage(L2), 3:ours')
    parser.add_argument('--T-size', type=int, default=10, help='the size of reparameterization metrix')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--n', type=int, default=100, help='number of items')
    parser.add_argument('--num-samples', type=int, default=0, help='number of samples, 0 -> all')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--seed', type=int, default=471298479,help='random seed')
    args = parser.parse_args()

    SEED = args.seed 
    print("Random seed: {}".format(SEED))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    portfolio_opt_dir = os.path.abspath(os.path.dirname(__file__))
    print("portfolio_opt_dir:", portfolio_opt_dir)

    sp500_data_dir = os.path.join(portfolio_opt_dir, "data", "sp500")
    sp500_data = SP500DataLoader(sp500_data_dir, "sp500",
                                 start_date=dt.datetime(2004, 1, 1),
                                 end_date=dt.datetime(2017, 1, 1),
                                 collapse="daily",
                                 overwrite=False,
                                 verbose=True)

    filepath = args.filepath
    n = args.n
    num_samples = args.num_samples if args.num_samples != 0 else 1000000
    num_epochs = args.epochs
    lr = args.lr
    method_id = args.method
    if method_id == 0:
        training_method = 'two-stage'
    elif method_id == 1:
        training_method = 'decision-focused'
    elif method_id == 2:
        training_method = 'two-stage2'
    elif method_id == 3:
        training_method = 'ours'
    else:
        raise ValueError('Not implemented methods')
    des = open("res/K100/"+training_method + "N"+str(args.n)+"newwithMSE_"+str(args.seed-471298479)+".txt", "w")
    print("Training method: {}".format(training_method))

    train_dataset, validate_dataset, test_dataset = generateDataset(sp500_data, n=n, num_samples=num_samples)
    feature_size = train_dataset.dataset[0][0].shape[1]

    model = PortfolioModel(input_size=feature_size, output_size=1, seed=args.seed)
    covariance_model = CovarianceModel(n=n, seed=args.seed)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(covariance_model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    if training_method == 'surrogate':
        T_size = args.T_size
        init_T = normalize_matrix_positive(torch.rand(n, T_size))
        T = torch.tensor(init_T, requires_grad=True)
        T_lr = lr
        T_optimizer = torch.optim.Adam([T], lr=T_lr)
        T_scheduler = ReduceLROnPlateau(T_optimizer, 'min')

    train_loss_list, train_obj_list = [], []
    test_loss_list,  test_obj_list  = [], []
    validate_loss_list,  validate_obj_list = [], []

    # generate C0, d0 and alpha0
    M1 = int(n * 0.4)
    C0 = sparsify_matrix(torch.ones(M1, n), sparsity=0.9)
    d0 = torch.from_numpy(np.random.random((M1, 1))) * 0.2
    alpha0 = torch.from_numpy(np.random.random((M1, 1)) * 0.3 * 50 / args.n)
    softcon = (C0, d0, alpha0)

    print('n: {}, lr: {}'.format(n,lr))
    print('Start training...')
    evaluate = True
    total_forward_time, total_inference_time, total_qp_time, total_backward_time = 0, 0, 0, 0
    forward_time_list, inference_time_list, qp_time_list, backward_time_list = [], [], [], []
    for epoch in range(-1, num_epochs):
        evaluate = True
        start_time = time.time()
        forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
        if training_method in ['decision-focused' ,'two-stage','two-stage2']:
            if epoch == -1:
                print('Testing the optimal solution...')
                train_loss, train_obj = test_portfolio(model, covariance_model, epoch, train_dataset, evaluate=True, softcon=(C0, d0, alpha0))
            elif epoch == 0:
                print('Testing the initial solution quality...')
                train_loss, train_obj = test_portfolio(model, covariance_model, epoch, train_dataset, evaluate=evaluate, softcon=(C0, d0, alpha0))
            else:
                train_loss, train_obj, (forward_time, inference_time, qp_time, backward_time) = train_portfolio(model, covariance_model, optimizer, epoch, train_dataset, training_method=training_method, evaluate=evaluate, softcon=softcon)
        elif training_method == "ours":
            if epoch == -1:
                print("Testing the optimal.")
                train_loss, train_obj = ours_test_portfolio(model, covariance_model, epoch, train_dataset, evaluate=True, softcon=(C0, d0, alpha0))
            elif epoch == 0:
                print("Testing the initial.")
                train_loss, train_obj = ours_test_portfolio(model, covariance_model, epoch, train_dataset, evaluate=evaluate, softcon=(C0, d0, alpha0))
            else:
                train_loss, train_obj, (forward_time, inference_time, qp_time, backward_time) = ours_train_portfolio(model, covariance_model, optimizer, epoch, train_dataset, evaluate=evaluate, softcon=softcon)

        else:
            raise ValueError('Not implemented')
        total_forward_time   += forward_time
        total_inference_time += inference_time
        total_qp_time        += qp_time
        total_backward_time  += backward_time

        forward_time_list.append(forward_time)
        inference_time_list.append(inference_time)
        qp_time_list.append(qp_time)
        backward_time_list.append(backward_time)

        # ================ validating ==================
        if training_method == "ours":
            if epoch == -1:
                validate_loss, validate_obj = ours_test_portfolio(model, covariance_model, epoch, validate_dataset, evaluate=True, softcon=(C0, d0, alpha0))
            else:
                validate_loss, validate_obj = ours_validate_portfolio(model, covariance_model, scheduler, epoch, validate_dataset, softcon=(C0, d0, alpha0))
        else:
            if epoch == -1:
                validate_loss, validate_obj = test_portfolio(model, covariance_model, epoch, validate_dataset, evaluate=True, softcon=(C0, d0, alpha0))
            else:
                validate_loss, validate_obj = validate_portfolio(model, covariance_model, scheduler, epoch, validate_dataset, training_method=training_method, evaluate=evaluate, softcon=(C0, d0, alpha0))

        # ================== testing ===================
        if training_method == "ours":
            if epoch == -1:
                test_loss, test_obj = ours_test_portfolio(model, covariance_model, epoch, test_dataset, evaluate=True, softcon=(C0, d0, alpha0))
            else:
                test_loss, test_obj = ours_test_portfolio(model, covariance_model, epoch, test_dataset, evaluate=evaluate, softcon=(C0, d0, alpha0))
        else:
            if epoch == -1:
                test_loss, test_obj = test_portfolio(model, covariance_model, epoch, test_dataset, evaluate=True, softcon=(C0, d0, alpha0))
            else:
                test_loss, test_obj = test_portfolio(model, covariance_model, epoch, test_dataset, evaluate=evaluate, softcon=(C0, d0, alpha0))

        # =============== printing data ================
        #sys.stdout.write(f'Epoch {epoch} | Train Loss:    \t {train_loss:.8f} \t | Train Objective Value:    \t {train_obj*100:.6f}% \n')
        #sys.stdout.write(f'Epoch {epoch} | Validate Loss: \t {validate_loss:.8f} \t | Validate Objective Value: \t {validate_obj*100:.6f}% \n')
        #sys.stdout.write(f'Epoch {epoch} | Test Loss:     \t {test_loss:.8f} \t | Test Objective Value:     \t {test_obj*100:.6f}% \n')
        #sys.stdout.flush()
        des.write('Epoch {} | Train Loss:    \t {} \t | Train Objective Value:    \t {}% \n'.format(epoch, train_loss, train_obj * 100))
        des.write('Epoch {} | Validate Loss: \t {} \t | Validate Objective Value: \t {}% \n'.format(epoch, validate_loss, validate_obj * 100))
        des.write('Epoch {} | Test Loss:     \t {} \t | Test Objective Value:     \t {}% \n'.format(epoch, test_loss, test_obj * 100))
        des.flush()
        # ============== recording data ================
        end_time = time.time()
        print("Epoch {}, elapsed time: {}, forward time: {}, inference time: {}, qp time: {}, backward time: {}".format(epoch, end_time - start_time, forward_time, inference_time, qp_time, backward_time))

        train_loss_list.append(train_loss)
        train_obj_list.append(train_obj)
        test_loss_list.append(test_loss)
        test_obj_list.append(test_obj)
        validate_loss_list.append(validate_loss)
        validate_obj_list.append(validate_obj)

        # record the data every epoch
        f_output = open('results/performance/' + filepath + "{}-SEED{}.csv".format(training_method,SEED), 'w')
        f_output.write('Epoch, {}\n'.format(epoch))
        f_output.write('training loss,' + ','.join([str(x) for x in train_loss_list]) + '\n')
        f_output.write('training obj,'  + ','.join([str(x) for x in train_obj_list])  + '\n')
        f_output.write('validating loss,' + ','.join([str(x) for x in validate_loss_list]) + '\n')
        f_output.write('validating obj,'  + ','.join([str(x) for x in validate_obj_list])  + '\n')
        f_output.write('testing loss,'  + ','.join([str(x) for x in test_loss_list])  + '\n')
        f_output.write('testing obj,'   + ','.join([str(x) for x in test_obj_list])   + '\n')
        f_output.close()

        f_time = open('results/time/' + filepath + "{}-SEED{}.csv".format(training_method, SEED), 'w')
        f_time.write('Epoch, {}\n'.format(epoch))
        f_time.write('Random seed, {}, forward time, {}, inference time, {}, qp time, {}, backward_time, {}\n'.format(str(seed), total_forward_time, total_inference_time, total_qp_time, total_backward_time))
        f_time.write('forward time,'   + ','.join([str(x) for x in forward_time_list]) + '\n')
        f_time.write('inference time,' + ','.join([str(x) for x in inference_time_list]) + '\n')
        f_time.write('qp time,'        + ','.join([str(x) for x in qp_time_list]) + '\n')
        f_time.write('backward time,'  + ','.join([str(x) for x in backward_time_list]) + '\n')
        f_time.close()
    des.close()
