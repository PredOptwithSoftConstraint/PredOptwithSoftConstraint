This repository is the code for NeurIPS 2021 submission "A Surrogate Objective Framework for Prediction+Programming with Soft Constraints".

Edit 2021/8/30: KKT-based (Decision-focused) baseline is added to the first experiment.

# Requirements
pytorch>=1.7.0

scipy

gurobipy (and Gurobi>=9.1 license - you can get Academic license for free at https://www.gurobi.com/downloads/end-user-license-agreement-academic/; download and install Gurobi first.)

Quandl

h5py

bs4

tqdm

sklearn 

pandas

lxml

qpth

cvxpy

cvxpylayers

# Running Experiments
You should be able to run all experiments by fulfilling the requirements and cloning this repo to your local machine.

## Synthetic Linear Programming
The dataset for this problem is generated at runtime. To run a single problem instance, type the following command:
```
python run_main_synth.py --method=2 --dim_context=40 --dim_hard=40 --dim_soft=20 --seed=2006 --dim_features=80 --loss=l1 --K=0.2
```
The four methods (L1,L2,SPO+,ours) we used in the experiment are respectively 
```
--method=0 --loss=l1 # L1
--method=0 --loss=l2 # L2
--method=1 --loss=l1 # SPO+
--method=2 --loss=l1 # ours
--method=3 --loss=l1 # decision-focused (KKT-based)
```
The other parameters can be seen in run_script.py and run_main_synth.py. To get multiple data for a single method, modify with the parameters listed above, and then run run_script.py. The outcome containing prediction error and regret is in the result folder. See dataprocess.py for a reference on how to interpret the data; the data with suffix "...test.txt" is used for evaluation. Also, to change batch size and training set size, alter the default parameters in run_main_synth.py.

## Portfolio Optimization
The dataset for this problem will be automatically downloaded when you first run this code, as Wilder et al.'s code does[1]. It is the daily price data of SP500 from 2004 to 2017 downloaded by Quandl API. To run a single problem instance, type the following command:
```
python main.py --method=3 --n=50 --seed=471298479
```
The four methods (L1, DF, L2, ours) are labeled as method 0, 1, 2 and 3. To get multiple data for a single method, run run_script.py.

The result is in the res/K100 folder.

## Resource Provisioning
The dataset of this problem is attached in the github repository, which are the eight csv file, one for each region. It is the ERCOT dataset taken from (...to be filled...), and is processed by resource_provisioning/data_energy/data_loader.py at runtime. When you first run this code, it will generate several large .npy file as the cached feature, which will accelerate the preprocessing of the following runs. This experiment requires large memory and is recommended to run on a server. To run a single problem instance, type the following command:
```
python run_main_newnet.py --method=1 --seed=16900000 --loss=l1
```
The four methods (L1, L2, weighted L1, ours) are respectively
```
--method=0 --loss=l1 # L1
--method=0 --loss=l2 # L2
--method=0 --loss=l3 # weighted L1
--method=1 --loss=l1 # ours
```
To run different ratio of alpha1/alpha2, modify line 157-158 in synthesize.py

```
 alpha1 = torch.ones(dim_context, 1) * 50
 alpha2 = torch.ones(dim_context, 1) * 0.5
```
to a desired ratio. Furthermore, modify line 174 in main_newnet.py
```
netname = "50to0.5"
```
to "5to0.5"/"1to1"/"0.5to5"/"0.5to50", and line 199 in main_newnet.py
```
self.alpha1, self.alpha2 = 0.5, 50
```
to (0.5, 5)/(1, 1)/(5, 0.5)/(50, 0.5) respectively.

run run_script.py to get multiple data. The result is in the result/2013to18_+str(netname)+newnet folder.  The interpretation of output data is similar to synthetic linear programming.

[1] Automatically Learning Compact Quality-aware Surrogates for Optimization Problems, Wilder et al., 2020 (https://arxiv.org/abs/2006.10815)

## Empirical Evaluation of Lambda_max in Theorem 6
run test.py directly to get results (note it takes a long time to finish the whole run, especially for the option of beta distribution). The results for uniform, Gaussian and beta are respectively in test1.txt, test2.txt and test3.txt.

<!--
**PredOptwithSoftConstraint/PredOptwithSoftConstraint** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
