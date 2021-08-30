import sys
import os
import argparse
import random
from config import device, modify_K
from util import *
from prednet_mo import PredNetMO
from calc import *
from spo import getopt_spo
from torch.optim import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax
import torch.nn as nn
from tqdm import tqdm
from synthesize import lowRankSynthesize,generateProblemParams
from DF import getopt_DF
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

# adaptors to calc.py
def getopt_adapt(theta, alpha, A, b, C, d):
    theta_ = theta.copy().reshape(-1, 1)
    return getopt(theta_, alpha, A, b, C, d)[1]


def getval_adapt(theta, x, alpha, A, b, C, d):
    x = x.reshape(-1, 1)
    theta_ = theta.copy().reshape(-1, 1)
    return getval(theta_, x, alpha, A, b, C, d)


def getopt_surro_adapt(theta, theta_pred_tensor, alpha, A, b, C, d):
    theta_ = torch.from_numpy(theta).to(device)
    theta_ = theta_.view(-1, 1)
    theta_pred_ = theta_pred_tensor.view(-1, 1)
    return getopt_surro(theta_, theta_pred_, alpha, A, b, C, d)


class PerfCompare:
    def __init__(self, alpha, A, b, C, d):
        self.alpha = alpha
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        
    def compare(self, theta_pred, theta, backlog=None, verb=None):
        """ theta: ground truth
        theta_pred: predicated theta
        """
        assert theta.shape == theta_pred.shape, "{a}!={b}: same shape required".format(a=theta.shape, b=theta_pred.shape)
        # regression metrics 
        mse = mean_squared_error(theta, theta_pred)
        mae = mean_absolute_error(theta, theta_pred)
        meae = median_absolute_error(theta, theta_pred)

        # reward metrics
        optvals = np.zeros(theta.shape[0])
        for i in range(optvals.shape[0]):
            x = getopt_adapt(theta[i], self.alpha, self.A, self.b, self.C, self.d)
            optvals[i] = getval_adapt(theta[i], x, self.alpha, self.A, self.b, self.C, self.d)
        vals = np.zeros(theta_pred.shape[0])
        for i in range(vals.shape[0]):
            x1 = getopt_adapt(theta_pred[i], self.alpha, self.A, self.b, self.C, self.d)
            vals[i] = getval_adapt(theta[i], x1, self.alpha, self.A, self.b, self.C, self.d)
        avg_optval = np.mean(optvals)
        avg_val = np.mean(vals)
        avg_reg = np.mean(optvals - vals) # mean of regrets
        if backlog is not None:
            backlog.write(str(mae)+" "+str(mse)+" "+str(meae)+" "+str(avg_optval)+" "+str(avg_val)+" "+str(avg_reg)+"\n")
            backlog.flush()
        if verb is not None:
            for i in range(vals.shape[0]):
                verb.write(str(vals[i])+" ")
            verb.write("\n")
            for i in range(optvals.shape[0]):
                verb.write(str(optvals[i])+" ")
            verb.write("\n")
            verb.flush()
        # print('mae',mae, 'mse',mse, 'meae',meae,'avg_optval',avg_optval, 'avg_val',avg_val, 'avg_reg',avg_reg)
        return {'mae':mae, 'mse':mse, 'meae':meae,'avg_optval':avg_optval, 'avg_val':avg_val, 'avg_reg':avg_reg}

def parse():
    parser = argparse.ArgumentParser(description="LPSoft-Syn")
    parser.add_argument("-d", "--directory", type=str, default='./', 
                        help="working directory")
    parser.add_argument("--seed", type=int, default=146358,
                        help='random seed')                         
    parser.add_argument("--method", type=int, default=3,
                        help='0=two-stage, 1=softLP, 2=SPO+, 3=DF')
    parser.add_argument("--num_two_stage_pretrain", type=int, default=40,
                        help='two-stage pretrain epoches')  
    parser.add_argument("--num_soft_training", type=int, default=40,
                        help='soft training epoches')
    parser.add_argument("--num_samples", type=int, default=150, # 150, 1500, 7500 train: 100, 1000, 5000 batch_size: 10, 50, 125
                        help="number of samples for predication")
    parser.add_argument("--dim_features", type=int, default=250,
                        help="dimension of features")
    parser.add_argument("--dim_latent", type=int, default=20,
                        help="dimension of latent var")
    parser.add_argument("--dim_context", type=int, default=2,
                        help="dimension of context (predication var)")
    parser.add_argument('--dim_decisions', type=int, default=2,
                        help="the number of decision vars")
    parser.add_argument('--dim_hard', type=int, default=10,
                        help="the number of hard limits")
    parser.add_argument('--dim_soft', type=int, default=1,
                        help="the number of (buyable) materials")               
    parser.add_argument("--loss", type=str, default='l1', 
                        help="loss function for prediction model")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="batch size of training")
    parser.add_argument("--max_norm", type=float, default=0.0001, 
                        help="max norm of grad")
    parser.add_argument("--init_lr", type=float, default=0.01,
                        help="initial learning rate")
    parser.add_argument("--K",type=float,default=QUAD_SOFT_K,
                        help="quad_soft_K")
    args = parser.parse_args()
    return args

valid_lst = []

def early_stopping(valid_lst):
    kk = 4
    GE_counts = np.sum(np.array(valid_lst[-kk:]) >= np.array(valid_lst[-2*kk:-kk]) + 1e-6)
    if GE_counts >= kk or np.sum(np.isnan(valid_lst[-kk:])) == kk:
        print("Early stopped at epoch", len(valid_lst))
        exit(0)

def main(args):    
    global buffer_C, buffer_d, buffer_alpha

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    
    # generate predication dataset (importance is deprecated)
    (xi_train, theta_train),(xi_valid, theta_valid), (xi_test, theta_test), importance = lowRankSynthesize(args.num_samples,
      args.dim_features, args.dim_context, args.dim_latent, feature_rank=args.dim_context, noise_level=0.01, seed=args.seed)

    # generate programming problems
    A0, b0, C0, d0, alpha0 = generateProblemParams(theta_train, args.dim_hard, args.dim_soft, importance)
    
    # add a performance compare
    pc = PerfCompare(alpha0, A0, b0, C0, d0)
    modify_K(args.K)
    print("QSK:", get_K())
    # method  
    if args.method == 0:
        training_method = 'two-stage'
    elif args.method == 1:
        training_method = 'soft-constraint'
    elif args.method == 2:
        training_method = 'SPO+'
    elif args.method == 3:
        training_method = 'DF'
    else:
        raise ValueError('Not implemented methods')
    # reset seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed((args.seed*17 + args.seed*args.seed*23) % (2**32))
    
    # build preditive model
    prednet = PredNetMO(args.dim_features, args.dim_context).to(device)
    #print(xi_train.shape, args.dim_features)
    optimizer = Adagrad(prednet.parameters(),lr=args.init_lr)#Adagrad(prednet.parameters(), lr=args.init_lr)
    print("featdim:", args.dim_features, "method:", training_method, "loss:", args.loss)
    des = open("result/"+training_method
               +"_size_"+str(args.dim_context)+"_"+str(args.dim_soft)+"_bs_"+str(args.batch_size)+"seed"+str(args.seed)+"loss"+str(args.loss)+"K"+str(args.K)+".txt", "w")
    des2 = open("result/" + training_method
               + "_size_" + str(args.dim_context) + "_" + str(args.dim_soft) + "_bs_" + str(args.batch_size) + "seed"+str(args.seed)+"loss"+str(args.loss)+"K"+str(args.K)+"test.txt",
               "w")
    des3 = open("result/"+training_method+"_size_"+str(args.dim_context)+"_"+str(args.dim_soft)+"_bs_"+str(args.batch_size)+"seed"+str(args.seed)+"loss"+str(args.loss)+"K"+str(args.K)+"valid.txt","w")
    verb = None #open("result/"+training_method+"_size_"+str(args.dim_context)+"_"+str(args.dim_soft)+"_bs_"+str(args.batch_size)+"seed"+str(args.seed)+"loss"+str(args.loss)+"K"+str(args.K)+"verbose.txt","w")
    verb2 = None #open("result/"+training_method+"_size_"+str(args.dim_context)+"_"+str(args.dim_soft)+"_bs_"+str(args.batch_size)+"seed"+str(args.seed)+"loss"+str(args.loss)+"K"+str(args.K)+"verbose_test.txt","w")
    verb3 = None #open("result/"+training_method+"_size_"+str(args.dim_context)+"_"+str(args.dim_soft)+"_bs_"+str(args.batch_size)+"seed"+str(args.seed)+"loss"+str(args.loss)+"K"+str(args.K)+"verbose_valid.txt","w")
    batch_size = args.batch_size
    max_norm = args.max_norm

    resetbuffer()
    repeat, will_eval = False, True
    print("theta_train:", theta_train)
    # two stage
    if training_method == "two-stage":
        fn_loss = nn.L1Loss() if args.loss=='l1' else nn.MSELoss()
        for i in tqdm(range(args.num_two_stage_pretrain)):
            if repeat: indices = np.array([i for i in range(xi_train.shape[0])])
            else: indices = np.random.permutation(xi_train.shape[0]) #shuffle
            batches = math.ceil(indices.shape[0]/batch_size)
            for bidx in range(batches):
                # train
                ## get batch data
                bs = min(batch_size, indices.shape[0] - batch_size*bidx)
                st = 0 if repeat else bidx
                idx = indices[batch_size*st :  batch_size*st+bs] 
                #print("idx:", idx)
                xi_batch = xi_train[idx]
                theta_batch = theta_train[idx]
                
                ## predict and backward
                theta_batch_pred = prednet(torch.from_numpy(xi_batch).to(device))
                loss = fn_loss(theta_batch_pred, torch.from_numpy(theta_batch).to(device))

                ## solve the progamming problems
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(prednet.parameters(), max_norm)
                optimizer.step()
 
                ## dump trainning perf comparison
                results = pc.compare(theta_batch_pred.detach().cpu().numpy(), theta_batch, des, verb)

                # test
                ## evaluate on test dataset

                ## dump test perf comparison
                if will_eval and bidx % batches == batches - 1:
                    theta_test_pred = prednet(torch.from_numpy(xi_test).to(device))
                    theta_valid_pred = prednet(torch.from_numpy(xi_valid).to(device))
                    results_valid = pc.compare(theta_valid_pred.detach().cpu().numpy(), theta_valid, des3, verb3)
                    results_test = pc.compare(theta_test_pred.detach().cpu().numpy(), theta_test, des2, verb2)
                    valid_lst.append(results_valid['avg_reg'])
                    if len(valid_lst) >= 8:
                        early_stopping(valid_lst)
                    #print("epoch {epoch},  batch {batch},  test perf: ".format(epoch=i, batch=bidx), results)

    # soft constraint
    elif training_method == "soft-constraint":
        # build new soft constraints
        mxtheta = np.maximum(np.max(theta_train, axis=0), np.max(theta_test,axis=0))
        mxtheta *= 2
        alpha, C, d = merge_constraints(A0, b0, C0, d0, alpha0, mxtheta)

        # train loop on all samples
        for i in tqdm(range(args.num_soft_training)):
            if repeat: indices = np.array([i for i in range(xi_train.shape[0])])
            else: indices = np.random.permutation(xi_train.shape[0]) #shuffle
            batches = math.ceil(indices.shape[0]/batch_size)
            for bidx in range(batches):
                # train
                ## get batch data
                bs = min(batch_size, indices.shape[0] - batch_size*bidx)
                st = 0 if repeat else bidx
                idx = indices[batch_size*st :  batch_size*st+bs] 
                #print("idx:",idx)
                xi_batch = xi_train[idx]
                theta_batch = theta_train[idx]

                ## predict
                theta_batch_pred = prednet(torch.from_numpy(xi_batch).to(device))

                ## solve the progamming problems
                loss = torch.zeros(1).to(device)
                for j in range(bs):
                    # keep theta_batch_pred as torch.tensor
                    val, grd = getopt_surro_adapt(theta_batch[j], theta_batch_pred[j], 
                                                  alpha0, A0, b0, C0, d0)
                    loss += -grd.view(1)
                loss /= bs #mean

                ## update grads
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(prednet.parameters(), max_norm) # 0.001
                optimizer.step()

                ## dump trainning perf comparison
                results = pc.compare(theta_batch_pred.detach().cpu().numpy(), theta_batch, des, verb)
                #print("epoch {epoch},  batch {batch},  train perf: ".format(epoch=i, batch=bidx), results)

                # test
                ## evaluate on test dataset

                ## dump test perf comparison
                if will_eval and bidx % batches == batches - 1:
                    theta_test_pred = prednet(torch.from_numpy(xi_test).to(device))
                    theta_valid_pred = prednet(torch.from_numpy(xi_valid).to(device))
                    results_test = pc.compare(theta_test_pred.detach().cpu().numpy(), theta_test, des2, verb2)
                    results_valid = pc.compare(theta_valid_pred.detach().cpu().numpy(), theta_valid, des3, verb3)
                    valid_lst.append(results_valid['avg_reg'])
                    if len(valid_lst) >= 8:
                        early_stopping(valid_lst)
                    #print("epoch {epoch},  batch {batch},  test perf: ".format(epoch=i, batch=bidx), results)

    elif training_method == "SPO+":
        # build new constraints

        A1 = torch.cat((torch.from_numpy(A0), torch.zeros(A0.shape[0], d0.shape[0])), dim=1)  # Ax <= b
        A2 = torch.cat((-torch.eye(args.dim_context),
                        torch.zeros(args.dim_context, d0.shape[0])), dim=1)  # -x <= 0
        A3 = torch.cat((torch.zeros(d0.shape[0], args.dim_context), -torch.eye(d0.shape[0])),
                       dim=1)  # -z <= 0
        A4 = torch.cat((torch.from_numpy(C0), -torch.eye(d0.shape[0])), dim=1)  # cx-z <= d
        A = torch.cat((A1, A2, A3, A4))
        b = torch.cat((torch.from_numpy(b0), torch.zeros(args.dim_context + d0.shape[0], 1),
                       torch.from_numpy(d0)))
        A, b = A.to(device), b.to(device)

        # train loop on all samples
        for i in tqdm(range(args.num_soft_training)):
            if repeat:
                indices = np.array([i for i in range(xi_train.shape[0])])
            else:
                indices = np.random.permutation(xi_train.shape[0])  # shuffle
            batches = math.ceil(indices.shape[0] / batch_size)
            for bidx in range(batches):
                # train
                ## get batch data
                bs = min(batch_size, indices.shape[0] - batch_size * bidx)
                st = 0 if repeat else bidx
                idx = indices[batch_size * st:  batch_size * st + bs]
                # print("idx:",idx)
                xi_batch = xi_train[idx]
                theta_batch = theta_train[idx]

                ## predict
                theta_batch_pred = prednet(torch.from_numpy(xi_batch).to(device))

                ## solve the progamming problems
                loss = torch.zeros(1).to(device)

                for j in range(bs):
                    # keep theta_batch_pred as torch.tensor

                    val, grd = getopt_spo(theta_batch[j], theta_batch_pred[j],
                                                  alpha0, A, b)
                    loss += -grd.view(1)
                #print(theta_batch_pred.shape, theta_batch.shape)
                loss /= bs  # mean
                optimizer.zero_grad()
                loss.backward()
                # print([torch.norm(param.grad) for param in prednet.parameters()])
                nn.utils.clip_grad_norm_(prednet.parameters(), max_norm)  # 0.001
                optimizer.step()

                # print("epoch {epoch},  batch {batch},  train perf: ".format(epoch=i, batch=bidx), results)

                ## dump test perf comparison
                if will_eval and bidx % batches == batches - 1:
                    theta_test_pred = prednet(torch.from_numpy(xi_test).to(device))
                    theta_valid_pred = prednet(torch.from_numpy(xi_valid).to(device))
                    results_test = pc.compare(theta_test_pred.detach().cpu().numpy(), theta_test, des2, verb2)
                    results_valid = pc.compare(theta_valid_pred.detach().cpu().numpy(), theta_valid, des3, verb3)
                    valid_lst.append(results_valid['avg_reg'])
                    if len(valid_lst) >= 8:
                        early_stopping(valid_lst)
                    # print("epoch {epoch},  batch {batch},  test perf: ".format(epoch=i, batch=bidx), results)

    # DF
    elif training_method == "DF":
        # train loop on all samples

        for i in tqdm(range(args.num_soft_training)):
            if repeat:
                indices = np.array([i for i in range(xi_train.shape[0])])
            else:
                indices = np.random.permutation(xi_train.shape[0])  # shuffle
            batches = math.ceil(indices.shape[0] / batch_size)
            for bidx in range(batches):
                # train
                ## get batch data
                bs = min(batch_size, indices.shape[0] - batch_size * bidx)
                st = 0 if repeat else bidx
                idx = indices[batch_size * st:  batch_size * st + bs]
                # print("idx:",idx)
                xi_batch = xi_train[idx]
                theta_batch = theta_train[idx]

                ## predict
                theta_batch_pred = prednet(torch.from_numpy(xi_batch).to(device))
                # print("pred:", theta_batch_pred)
                ## solve the programming problems
                loss = torch.zeros(1).to(device)
                for j in range(bs):
                    # keep theta_batch_pred as torch.tensor
                    val, grd = getopt_DF(theta_batch[j], theta_batch_pred[j],
                                                  alpha0, A0, b0, C0, d0)
                    loss += -grd.view(1)
                loss /= bs  # mean

                ## update grads
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(prednet.parameters(), max_norm)  # 0.001
                optimizer.step()

                ## dump trainning perf comparison
                results = pc.compare(theta_batch_pred.detach().cpu().numpy(), theta_batch, des, verb)
                # print("epoch {epoch},  batch {batch},  train perf: ".format(epoch=i, batch=bidx), results)

                # test
                ## evaluate on test dataset
                # print("alpha:", alpha0)
                ## dump test perf comparison
                if will_eval and bidx % batches == batches - 1:
                    theta_test_pred = prednet(torch.from_numpy(xi_test).to(device))
                    theta_valid_pred = prednet(torch.from_numpy(xi_valid).to(device))
                    results_test = pc.compare(theta_test_pred.detach().cpu().numpy(), theta_test, des2, verb2)
                    results_valid = pc.compare(theta_valid_pred.detach().cpu().numpy(), theta_valid, des3, verb3)
                    valid_lst.append(results_valid['avg_reg'])
                    if len(valid_lst) >= 8:
                        early_stopping(valid_lst)
                    # print("epoch {epoch},  batch {batch},  test perf: ".format(epoch=i, batch=bidx), results)

if __name__ == "__main__":
    """ solve synthetic problems
    """
    args = parse()
    sys.exit(main(args))

