# script for multiple seeds.
import os
for seed in range(4, 5):
    for N in [250]:
        os.system("python main.py --method=3 --n="+str(N)+" --seed="+str(seed + 471298479))
