# for theta prediction normalization; deprecated and permanently set to 1.
THETA_COMPRESSED_RATIO = 1
# for quad soft surrogate
QUAD_SOFT_K = 100 # The hyperparameter K
SURROOFSURRO = 0.0001
# for sigmoid soft surrogate
CLIP = "NOCLIP" # gradient clip option: noclip, clip or normalize. Deprecated and permanently set to "NOCLIP";
# the clipping work is done by pytorch.
USE_L1_LOSS = True # for two-stage method. This configuration only controls surrotest/main.py.
PRETRAIN_TAG = False # deprecated.
DROPOUT = False
OUTPUT_NAME = "cplus" # folder for outputting debug message.
RATIO = 1 # deprecated and permanently set to 1.
