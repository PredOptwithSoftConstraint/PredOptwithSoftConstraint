import numpy as np
import scipy.linalg
import random
import numpy.random
from tqdm import tqdm
method = 3 # 1, 2, 3
SEED = 29489138
random.seed(SEED)
np.random.seed(SEED)
def getnormal(n):
    if method == 1:
        return np.random.rand(n, 1) - 0.5
    elif method == 2:
        return np.random.normal(size=(n, 1))
    else:
        return np.random.beta(2, 2, size=(n, 1)) - 0.5
def getpositive(n):
    if method == 1:
        return np.random.rand(n)
    elif method == 2:
        return np.abs(np.random.normal(size=n))
    else:
        return np.random.beta(2, 2, size=n)

f = open("test"+str(method)+".txt", "w")
for n in tqdm(range(5, 300, 5)):
    tests = 1000
    A = np.zeros((n, n))
    S = np.zeros(tests)
    for T in range(tests):
        normal = getnormal(n)#np.random.beta(2, 2, size=(n, 1)) - 0.5 #np.random.normal(size=(n, 1))# np.random.rand(n, 1) - 0.5
        # print(normal)
        for i in range(n):
            seed = random.random()
            ctr = 0
            while True:
                ctr += 1
                if ctr >= 10000: break
                if seed < 0.5:  # negative part
                    A[i] = -getpositive(n) # -np.abs(np.random.normal(size=n)) #-np.random.rand(n)
                    # print("value negative:", np.dot(A[i].reshape(1, A[i].shape[0]), normal))
                    if np.dot(A[i].reshape(1, A[i].shape[0]), normal) < 0:
                        continue
                    else:
                        A[i] /= np.linalg.norm(A[i])
                        break
                else:  # positive part
                    A[i] = getpositive(n) # np.abs(np.random.normal(size=n))#np.random.rand(n)
                    # print("value positive:", np.dot(A[i].reshape(1, A[i].shape[0]), normal))
                    if np.dot(A[i].reshape(1, A[i].shape[0]), normal) < 0:
                        continue
                    else:
                        A[i] /= np.linalg.norm(A[i])
                        break
        # print(A)
        R, Q = scipy.linalg.rq(A)
        # print(R, Q)
        V = np.matmul(R.T, R)
        S[T] = np.linalg.eig(V)[0][0]
        # print(T, S[:T + 1], np.mean(S[:T + 1]), np.std(S[:T + 1], ddof=1), np.max(S[:, T+1]))
    f.write(str(n)+" "+str(np.mean(S[:T + 1])) + " " + str(np.std(S[:T + 1], ddof=1)) + " "+str(np.max(S[:T+1]))+"\n")
    f.flush()
f.close()
