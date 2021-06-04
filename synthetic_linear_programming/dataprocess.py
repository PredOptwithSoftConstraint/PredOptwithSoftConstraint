import numpy as np
import matplotlib.pyplot as plt
methods = [("two-stage", "l1"), ("two-stage", "l2"), ("SPO+", "l1"), ("soft-constraint", "l1")]

# methods = [("soft-constraint", "l1")]
SIZE, M, N = (80, 40), 15, 40
BS = 125
seed = 2006
oursK = 0.2
perf = np.zeros((len(methods), M))

def read(f, is_other="no"):
    lines = f.readlines()
    lst = []
    for line in lines:
        contents = line.split()
        if is_other == "MAE":
            lst.append(float(contents[0]))
        elif is_other == "MSE":
            lst.append(float(contents[1]))
        else:
            lst.append(float(contents[5]))# 0 = MAE, 1 = MSE, 2 = MEDIAN ABS E, 3 = OPT, 4 = VAL, 5 = REG
    return lst

for k, method in enumerate(methods):
    for i in range(M):
        if method[0] == "soft-constraint":
            name = method[0] + "_size_" + str(SIZE[0]) + "_" + str(SIZE[1]) + "_bs_" + str(BS) + "seed" + str(seed + i) + "loss"+method[1]+"K"+str(oursK)
        else:
            name = method[0] + "_size_" + str(SIZE[0]) + "_" + str(SIZE[1]) + "_bs_" + str(BS) + "seed" + str(seed + i) + "loss"+method[1]+"K0.2"
        f = open(name+"valid.txt", "r")
        g = open(name+"test.txt", "r")
        is_other = "no"
        #if method[0] == "two-stage":
        #    is_other = "MAE" if method[1] == "l1" else "MSE"
        valid_lst = read(f, is_other)
        test_lst = read(g)
        if len(test_lst) < N:
            print("Early Stopping!", method, len(valid_lst), len(test_lst))
        kk = 4
        # print(method, ":", valid_lst)
        vall = []
        stop_flag = False
        for epoch in range(N):
            print(epoch, name)
            vall.append(valid_lst[epoch])
            if epoch >= kk * 2 - 1:
                if method[0] == 'two-stage':
                    print(vall[-kk:], vall[-2*kk:-kk])
                    GE_counts = np.sum(np.array(vall[-kk:]) >= np.array(vall[-2 * kk:-kk]) + 1e-6)
                    print('Generalization error increases counts: {}'.format(GE_counts))
                    if GE_counts >= kk or np.sum(np.isnan(vall[-kk:])) == kk:
                        print(method, epoch, ":", test_lst[epoch])
                        perf[k, i] = test_lst[epoch]
                        stop_flag = True
                        break
                else:  # surrogate or decision-focused
                    GE_counts = np.sum(np.array(vall[-kk:]) >= np.array(vall[-2 * kk:-kk]) + 1e-6)
                    print('Generalization error increases counts: {}'.format(GE_counts))
                    if GE_counts >= kk or np.sum(np.isnan(vall[-kk:])) == kk:
                        print(method, epoch, ":", test_lst[epoch])
                        perf[k, i] = test_lst[epoch]
                        stop_flag = True
                        break
        if not stop_flag:
            print(vall)
            print(test_lst)
            perf[k, i] = test_lst[-1]
        #plt.plot([i for i in range(len(valid_lst))], valid_lst)
        #plt.show()
print(np.mean(perf, axis=1))
print(np.std(perf, axis=1, ddof=1))
