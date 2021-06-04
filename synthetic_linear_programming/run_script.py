import os
size = [40, 40, 80, 80]
hard = [40, 40, 80, 80]
soft = [1, 20, 1, 40]
features = [80, 80, 160, 160]
K = [0.2, 1, 5, 25, 125]
for j in range(4):
    for i in range(10, 15):
        for k in range(1):
            os.system("python run_main_synth.py --method=2 --dim_context="+str(size[j])+" --dim_hard="+str(hard[j])+" --dim_soft="+str(soft[j])+" --seed="+str(i+2006)+" --dim_features="+str(features[j])+" --loss=l1 --K="+str(K[k]))
        # os.system("python run_main_synth.py --method=1 --dim_context="+str(size[j])+" --dim_hard="+str(hard[j])+" --dim_soft="+str(soft[j])+" --seed="+str((i * 123498129 + j * 1324879 + 129587912345)%(2 ** 32))) #W32

