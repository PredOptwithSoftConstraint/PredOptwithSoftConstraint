import sys
import os
import argparse
import random
from config import device
from util import *
from prednet_energy import PredNet3D
from calc import *
from data_energy.data_loader import REGIONS
from data_loader_energy import make_dataset
from torch.optim import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax
import torch.nn as nn
from tqdm import tqdm
from synthesize import lowRankSynthesize,generateProblemParams
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error


# adaptors to calc.py
def getopt_adapt(alpha1, alpha2, A, b, C, d):
    return getopt(alpha1, alpha2, A, b, C, d)[1]


def getval_adapt(x, alpha0, alpha1, A, b, C, d):
    x = x.reshape(-1, 1)
    return getval_twoalpha(x, alpha0, alpha1, A, b, C, d)


def readData(seed, region, N):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    X0, y0 = make_dataset(region, N)
    y0 /= 10000 # for normalization
    # X0 shape: min(148896, N) * 24 * 77
    # y0 shape: min(148896, N) * 24
    return X0, y0

class PerfCompare:
    def __init__(self, alpha1, alpha2, A, b, d):
        self.alpha1, self.alpha2 = alpha1, alpha2
        self.A = A
        self.b = b
        self.d = d
        
    def compare(self, C_pred, C, backlog=None, verb=None, special_input=None):
        """ theta: ground truth
        theta_pred: predicated theta
        """
        assert C.shape == C_pred.shape, "{a}!={b}: same shape required".format(a=C.shape, b=C_pred.shape)
        # regression metrics 
        mse = mean_squared_error(C.reshape(C.shape[0], -1), C_pred.reshape(C_pred.shape[0], -1))
        mae = mean_absolute_error(C.reshape(C.shape[0], -1), C_pred.reshape(C_pred.shape[0], -1))
        meae = median_absolute_error(C.reshape(C.shape[0], -1), C_pred.reshape(C_pred.shape[0], -1))

        # reward metrics
        optvals = np.zeros(C.shape[0])
        for i in range(optvals.shape[0]):
            x = getopt_adapt(self.alpha1, self.alpha2, self.A, self.b, C[i], self.d)
            optvals[i] = getval_adapt(x, self.alpha1, self.alpha2, self.A, self.b, C[i], self.d)
        vals = np.zeros(C_pred.shape[0])
        for i in range(vals.shape[0]):
            if special_input is None: x1 = getopt_adapt(self.alpha1, self.alpha2, self.A, self.b, C_pred[i], self.d)
            else: x1 = special_input[i] # for amos
            vals[i] = getval_adapt(x1, self.alpha1, self.alpha2, self.A, self.b, C[i], self.d)
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
    parser.add_argument("--seed", type=int, default=146370,
                        help='random seed')                         
    parser.add_argument("--method", type=int, default=1,
                        help='0=two-stage, 1=softLP, 2=QPTL')
    parser.add_argument("--num_two_stage_pretrain", type=int, default=6,
                        help='two-stage pretrain epoches')  
    parser.add_argument("--num_soft_training", type=int, default=6,
                        help='soft training epoches')
    parser.add_argument("--num_samples", type=int, default=100,
                        help="number of samples for predication")
    parser.add_argument("--dim_features", type=int, default=24 * 77, # fixed for now.
                        help="dimension of features")
    parser.add_argument("--dim_latent", type=int, default=20, # deprecated for now
                        help="dimension of latent var")
    parser.add_argument("--dim_context", type=int, default=24, # fixed for now
                        help="dimension of context (predication var)")
    parser.add_argument('--dim_decisions', type=int, default=20,
                        help="the number of decision vars")
    parser.add_argument('--dim_hard', type=int, default=20,
                        help="the number of hard limits")
    parser.add_argument('--dim_soft', type=int, default=8, # fixed for now
                        help="the number of (buyable) materials")               
    parser.add_argument("--loss", type=str, default='l2',
                        help="loss function for prediction model")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size of training")
    parser.add_argument("--max_norm", type=float, default=0.01,
                        help="max norm of grad")
    parser.add_argument("--init_lr", type=float, default=0.01,
                        help="initial learning rate")
    args = parser.parse_args()
    assert args.dim_soft == len(REGIONS), "DIM_SOFT ERROR!"
    return args



def main(args):    
    global buffer_C, buffer_d, buffer_alpha

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    
    # generate predication dataset
    X0s, y0s = [], []
    N = 60000 - 7465 # start from year 2013
    for region in REGIONS:
        print("region:", region)
        X0, y0 = readData(args.seed, region, N + 24) # not full dataset
        print(X0.shape, y0.shape)
        X0s.append(X0.reshape(tuple([1] + list(X0.shape)))), y0s.append(y0.reshape(tuple([1] + list(y0.shape))))
        del X0, y0
    X0 = np.concatenate(X0s, axis=0).transpose((1, 0, 2, 3))
    #X0 = X0.reshape(X0.shape[0], X0.shape[1], -1)
    del X0s
    y0 = np.concatenate(y0s, axis=0).transpose((1, 2, 0))
    del y0s
    train, valid, test = [i for i in range(int(N * 0.7))], [i for i in range(int(N * 0.7), int(N * 0.8))], [i for i in range(int(N * 0.8), N)]
    # print(train[-1], test[-1])
    xi_train, xi_valid, xi_test = X0[train, :, :], X0[valid, :, :], X0[test, :, :] # N * 8 * (24*77)
    del X0
    C_train, C_valid, C_test = y0[train, :, :], y0[valid, :, :], y0[test, :, :] # N * 24 * 8
    del y0
    # generate programming problems
    A0, b0, d0, alpha1, alpha2 = generateProblemParams(args.dim_context, args.dim_hard, args.dim_soft, seed=args.seed)
    
    # add a performance compare
    pc = PerfCompare(alpha1, alpha2, A0, b0, d0)
    # alpha 1 is the coeff in the front; alpha 2 is in the rear.
    # method  
    if args.method == 0:
        training_method = 'two-stage'
    elif args.method == 1:
        training_method = 'soft-constraint'
    elif args.method == 2:
        training_method = 'Amos'
    else:
        raise ValueError('Not implemented methods')
    # reset seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed((args.seed*17 + args.seed*args.seed*23) % (2**32))
    
    # build preditive model
    prednet = PredNet3D((24, 77), (24,), seed=args.seed).to(device)
    #print(xi_train.shape, args.dim_features)
    #exit(0)
    optimizer = Adagrad(prednet.parameters(),lr=args.init_lr)
    netname = "50to0.5"
    print("featdim:", args.dim_features, "method:", training_method, "loss:", args.loss)
    des = open("result/2013to18_"+netname+"_newnet/"+training_method
               +"_size_"+str(args.dim_context)+"_"+str(args.dim_hard)+"_bs_"+str(args.batch_size)+"seed"+str(args.seed)+"loss"+str(args.loss)+"K"+str(QUAD_SOFT_K)+".txt", "w")
    des2 = open("result/2013to18_"+netname+"_newnet/" + training_method
               + "_size_" + str(args.dim_context) + "_" + str(args.dim_hard) + "_bs_" + str(args.batch_size) + "seed"+str(args.seed)+"loss"+str(args.loss)+"K"+str(QUAD_SOFT_K)+"test.txt",
               "w")
    des3 = open("result/2013to18_"+netname+"_newnet/" + training_method
               + "_size_" + str(args.dim_context) + "_" + str(args.dim_hard) + "_bs_" + str(args.batch_size) + "seed"+str(args.seed)+"loss"+str(args.loss)+"K"+str(QUAD_SOFT_K)+"valid.txt",
               "w")
    verb = open("result/2013to18_"+netname+"_newnet/"+training_method+"_size_"+str(args.dim_context)+"_"+str(args.dim_hard)+"_bs_"+str(args.batch_size)+"seed"+str(args.seed)+"loss"+str(args.loss)+"K"+str(QUAD_SOFT_K)+"verbose.txt","w")
    verb2 = open("result/2013to18_"+netname+"_newnet/"+training_method+"_size_"+str(args.dim_context)+"_"+str(args.dim_hard)+"_bs_"+str(args.batch_size)+"seed"+str(args.seed)+"loss"+str(args.loss)+"K"+str(QUAD_SOFT_K)+"verbose_test.txt","w")
    verb3 = open("result/2013to18_"+netname+"_newnet/"+training_method+"_size_"+str(args.dim_context)+"_"+str(args.dim_hard)+"_bs_"+str(args.batch_size)+"seed"+str(args.seed)+"loss"+str(args.loss)+"K"+str(QUAD_SOFT_K)+"verbose_valid.txt","w")

    batch_size = args.batch_size
    max_norm = args.max_norm
 
    resetbuffer()
    repeat, will_eval = False, True
    # two stage
    if training_method == "two-stage":
        #fn_loss = nn.L1Loss() if args.loss=='l1' else nn.MSELoss()
        class weighted_l1loss(nn.Module):
            def __init__(self):
                super(weighted_l1loss, self).__init__()
                self.alpha1, self.alpha2 = 0.5, 50
            def forward(self, x, y): # x is pred, y is label
                return self.alpha1 * torch.sum(torch.clamp(x - y, min=0)) + self.alpha2 * torch.sum(torch.clamp(y - x, min=0))
        if args.loss == "l1": fn_loss = nn.L1Loss()
        elif args.loss == "l2": fn_loss = nn.MSELoss()
        else: 
            fn_loss = weighted_l1loss()
            print("using weighted loss!")
        for i in tqdm(range(args.num_two_stage_pretrain)):
            if repeat: indices = np.array([i for i in range(xi_train.shape[0])])
            else: indices = np.random.permutation(xi_train.shape[0]) #shuffle
            batches = math.ceil(indices.shape[0]/batch_size)
            for bidx in tqdm(range(batches)):
                # train
                ## get batch data
                bs = min(batch_size, indices.shape[0] - batch_size*bidx)
                st = 0 if repeat else bidx
                idx = indices[batch_size*st :  batch_size*st+bs] 
                #print("idx:", idx)
                xi_batch = xi_train[idx] # idx * 8 * 24 * 77
                C_batch = C_train[idx]   # idx * 24 * 8
                ## predict and backward
                C_batch_pred = torch.zeros(xi_batch.shape[0], args.dim_context, args.dim_soft)  # dim_soft = 8, dim_context = 24
                for i in range(xi_batch.shape[0]):
                    C_batch_pred[i, :, :] = prednet(torch.from_numpy(xi_batch[i]).to(device)).t()
                loss = fn_loss(C_batch_pred, torch.from_numpy(C_batch).to(device))

                ## solve the progamming problems
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(prednet.parameters(), max_norm)
                optimizer.step()
 
                ## dump trainning perf comparison
                results = pc.compare(C_batch_pred.detach().cpu().numpy(), C_batch, des, verb)
                #print("epoch {epoch},  batch {batch},  train perf: ".format(epoch=i, batch=bidx),results)

                # test
                ## evaluate on test dataset

                ## dump test perf comparison
                if will_eval and bidx in [batches // 4, (batches * 2) // 4, (batches * 3) // 4, batches - 1]: # output every 10 batch
                    C_test_pred = torch.zeros(xi_test.shape[0], args.dim_context, args.dim_soft)  # dim_soft = 8, dim_context = 24
                    C_valid_pred = torch.zeros(xi_valid.shape[0], args.dim_context, args.dim_soft)
                    with torch.no_grad():
                        for i in range(xi_test.shape[0]):
                            C_test_pred[i, :, :] = prednet(torch.from_numpy(xi_test[i]).to(device)).t()
                        for i in range(xi_valid.shape[0]):
                            C_valid_pred[i, :, :] = prednet(torch.from_numpy(xi_test[i]).to(device)).t()
                    _ = pc.compare(C_valid_pred.cpu().numpy(), C_valid, des3, verb3)
                    results = pc.compare(C_test_pred.cpu().numpy(), C_test, des2, verb2)
                    del C_test_pred, C_valid_pred
                    #print("epoch {epoch},  batch {batch},  test perf: ".format(epoch=i, batch=bidx), results)

    # soft constraint
    elif training_method == "soft-constraint":
        # train loop on all samples 
        for i in tqdm(range(args.num_soft_training)):
            if repeat: indices = np.array([i for i in range(xi_train.shape[0])])
            else: indices = np.random.permutation(xi_train.shape[0]) #shuffle
            batches = math.ceil(indices.shape[0]/batch_size)
            for bidx in tqdm(range(batches)):
                # train
                ## get batch data
                bs = min(batch_size, indices.shape[0] - batch_size*bidx)
                st = 0 if repeat else bidx
                idx = indices[batch_size*st :  batch_size*st+bs] 
                #print("idx:",idx)
                xi_batch = xi_train[idx]
                C_batch = C_train[idx]

                ## predict
                C_batch_pred = torch.zeros(xi_batch.shape[0], args.dim_context, args.dim_soft)  # dim_soft = 8, dim_context = 24
                for i in range(xi_batch.shape[0]):
                    C_batch_pred[i, :, :] = prednet(torch.from_numpy(xi_batch[i]).to(device)).t()
                # print(C_batch_pred.shape, C_batch.shape)
                ## solve the progamming problems
                loss = torch.zeros(1).to(device)
                for j in range(bs):
                    # keep theta_batch_pred as torch.tensor
                    val, grd = getopt_surro(C_batch[j], C_batch_pred[j],
                                                  alpha1, alpha2, A0, b0, d0)
                    loss += -grd.view(1)
                loss /= bs #mean

                ## update grads
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(prednet.parameters(), max_norm) # 0.001
                optimizer.step()

                ## dump trainning perf comparison
                results_train = pc.compare(C_batch_pred.detach().cpu().numpy(), C_batch, des, verb)
                #print("epoch {epoch},  batch {batch},  train perf: ".format(epoch=i, batch=bidx), results)

                # test
                ## evaluate on test dataset
                ## dump test perf comparison
                if will_eval and bidx in [batches // 4, (batches * 2) // 4, (batches * 3) // 4, batches - 1]: # output every 10 batch
                    C_test_pred = torch.zeros(xi_test.shape[0], args.dim_context, args.dim_soft)  # dim_soft = 8, dim_context = 24
                    C_valid_pred = torch.zeros(xi_valid.shape[0], args.dim_context, args.dim_soft)
                    with torch.no_grad():
                        for i in range(xi_test.shape[0]):
                            C_test_pred[i, :, :] = prednet(torch.from_numpy(xi_test[i]).to(device)).t()
                        for i in range(xi_valid.shape[0]):
                            C_valid_pred[i, :, :] = prednet(torch.from_numpy(xi_test[i]).to(device)).t()
                    _ = pc.compare(C_valid_pred.cpu().numpy(), C_valid, des3, verb3)
                    results = pc.compare(C_test_pred.cpu().numpy(), C_test, des2, verb2)
                    del C_test_pred, C_valid_pred
                    #print("epoch {epoch},  batch {batch},  test perf: ".format(epoch=i, batch=bidx), results)
    torch.save(prednet, "models/"+netname+"_"+training_method+"_"+args.loss+str(args.seed)+".pth")
    
if __name__ == "__main__":
    args = parse()
    sys.exit(main(args))

