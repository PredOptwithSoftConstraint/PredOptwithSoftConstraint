import os
for i in range(10,15):
    os.system("python run_main_newnet.py --method=1 --seed="+str(i+16900000)+" --loss=l1")
        # os.system("python run_main_synth.py --method=1 --dim_context="+str(size[j])+" --dim_hard="+str(hard[j])+" --dim_soft="+str(soft[j])+" --seed="+str((i * 123498129 + j * 1324879 + 129587912345)%(2 ** 32))) #W32

