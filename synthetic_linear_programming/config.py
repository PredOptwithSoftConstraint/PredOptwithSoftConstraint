import os
import torch
import numpy as np
# for theta prediction normalization. deprecated.
THETA_COMPRESSED_RATIO = 1
# for quad soft surrogate
QUAD_SOFT_K = 5 # 20 by default
SURROOFSURRO = 0.0001
# for sigmoid soft surrogate
grad_change = 0.05 # 10x quicker than normal sigmoid FIXME
epsilon_1 = 0.001  # the upper bound
epsilon_2 = -0.001  # the lower bound
CLIP = "NOCLIP"
USE_L1_LOSS = True # for two-stage method. This configuration only controls surrotest/main.py.
PRETRAIN_TAG = False
DROPOUT = False

def get_best_gpu(force = None):
    if force is not None:return force
    s = os.popen("nvidia-smi --query-gpu=memory.free --format=csv")
    a = []
    ss = s.read().replace('MiB','').replace('memory.free','').split('\n')
    s.close()
    for i in range(1, len(ss) - 1):
        a.append(int(ss[i]))
    print(a)
    best = int(np.argmax(a))
    print('the best GPU is ',best,' with free memories of ',ss[best + 1])
    return best

def modify_K(val):
    global QUAD_SOFT_K
    QUAD_SOFT_K = val

def get_K():
    return QUAD_SOFT_K

device = torch.device("cuda:"+str(get_best_gpu())) if torch.cuda.is_available() else torch.device("cpu")
