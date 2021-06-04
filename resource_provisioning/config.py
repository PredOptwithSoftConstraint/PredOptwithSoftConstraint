import os
import torch
import numpy as np
# for theta prediction normalization. deprecated.
THETA_COMPRESSED_RATIO = 1
# for quad soft surrogate
QUAD_SOFT_K = 0.05
SURROOFSURRO = 0.0001 # deprecated.
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


device = torch.device("cuda:"+str(get_best_gpu())) if torch.cuda.is_available() else torch.device("cpu")
print(device)
