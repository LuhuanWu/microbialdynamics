import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

from src.distribution.mvn import tf_mvn
from src.model import SSM

flags = tf.app.flags

Dx = 2
Dy = 3
Dv = 2
Dev = 0

q0_layers = [16, 16]        # q(x_1|y_1) or q(x_1|y_1:T)
q1_layers = [16, 16]        # q(x_t|x_{t-1}), including backward evolution term q(x_{t-1}|x_t)
q2_layers = [16, 16]        # q(x_t|y_t) or q(x_t|y_1:T)
f_layers = [16, 16]         # target evolution
g_layers = [16, 16]         # target emission

# Covariance Terms
q0_sigma_init, q0_sigma_min = 5, 1
q1_sigma_init, q1_sigma_min = 5, 1
q2_sigma_init, q2_sigma_min = 5, 1
f_sigma_init, f_sigma_min = 5, 1
g_sigma_init, g_sigma_min = 5, 1

# if q, f and g networks also output covariance (sigma)
output_cov = False

# whether the networks only output diagonal value of cov matrix
diag_cov = False


# bidirectional RNN, number of units in each LSTM cells
# For example, [32, 32] means a bRNN composed of 2 LSTM cells, 32 units in each cell
y_smoother_Dhs = [32]
X0_smoother_Dhs = [32]

# whether use a separate RNN for getting X0
X0_use_separate_RNN = True

# whether use tf.contrib.rnn.stack_bidirectional_dynamic_rnn or tf.nn.bidirectional_dynamic_rnn
# check https://stackoverflow.com/a/50552539 for differences between them
use_stack_rnn = True

# whether emission uses Dirichlet distribution
dirichlet_emission = True

# whether q1 (evolution term in proposal) and f share the same network
# (ATTENTION: even if use_2_q == True, f and q1 can still use different networks)
use_bootstrap = True

# should q use true_X to sample? (useful for debugging)
q_uses_true_X = False

# if q uses two networks q1(x_t|x_t-1) and q2(x_t|y_t)
# if True, q_uses_true_X will be overwritten as False
use_2_q = True

# whether emission uses Poisson distribution
poisson_emission = False

# ------------------------- Inference Schemes ------------------------ #
# Choose one of the following objectives
PSVO = False      # Particle Smoothing Variational Objective (use Forward Filtering Backward Simulation)
SVO = False      # Smoothing Variational Objective (use proposal based on bRNN)
AESMC = True    # Auto-Encoding Sequential Monte Carlo
IWAE = False     # Importance Weighted Auto-Encoder

BSim_use_single_RNN = False

q0_layers = ",".join([str(x) for x in q0_layers])
q1_layers = ",".join([str(x) for x in q1_layers])
q2_layers = ",".join([str(x) for x in q2_layers])
f_layers = ",".join([str(x) for x in f_layers])
g_layers = ",".join([str(x) for x in g_layers])
y_smoother_Dhs = ",".join([str(x) for x in y_smoother_Dhs])
X0_smoother_Dhs = ",".join([str(x) for x in X0_smoother_Dhs])

batch_size = 1

flags.DEFINE_integer("Dx", Dx, "dimension of hidden states")
flags.DEFINE_integer("Dy", Dy, "dimension of observations")
flags.DEFINE_integer("Dv", Dv, "dimension of inputs")
flags.DEFINE_integer("Dev", Dev, "input embedding size")

flags.DEFINE_string("q0_layers", q0_layers, "architecture for q0 network, int seperated by comma, "
                                            "for example: '50,50' ")
flags.DEFINE_string("q1_layers", q1_layers, "architecture for q1 network, int seperated by comma, "
                                            "for example: '50,50' ")
flags.DEFINE_string("q2_layers", q2_layers, "architecture for q2 network, int seperated by comma, "
                                            "for example: '50,50' ")
flags.DEFINE_string("f_layers",  f_layers,  "architecture for f network, int seperated by comma, "
                                            "for example: '50,50' ")
flags.DEFINE_string("g_layers",  g_layers,  "architecture for g network, int seperated by comma, "
                                            "for example: '50,50' ")

flags.DEFINE_float("q0_sigma_init", q0_sigma_init, "initial value of q0_sigma")
flags.DEFINE_float("q1_sigma_init", q1_sigma_init, "initial value of q1_sigma")
flags.DEFINE_float("q2_sigma_init", q2_sigma_init, "initial value of q2_sigma")
flags.DEFINE_float("f_sigma_init",  f_sigma_init,  "initial value of f_sigma")
flags.DEFINE_float("g_sigma_init",  g_sigma_init,  "initial value of g_sigma")

flags.DEFINE_float("q0_sigma_min", q0_sigma_min, "minimal value of q0_sigma")
flags.DEFINE_float("q1_sigma_min", q1_sigma_min, "minimal value of q1_sigma")
flags.DEFINE_float("q2_sigma_min", q2_sigma_min, "minimal value of q2_sigma")
flags.DEFINE_float("f_sigma_min",  f_sigma_min,  "minimal value of f_sigma")
flags.DEFINE_float("g_sigma_min",  g_sigma_min,  "minimal value of g_sigma")

flags.DEFINE_boolean("output_cov", output_cov, "whether q, f and g networks also output covariance (sigma)")
flags.DEFINE_boolean("diag_cov", diag_cov, "whether the networks only output diagonal value of cov matrix")

# bidirectional RNN
flags.DEFINE_string("y_smoother_Dhs", y_smoother_Dhs, "number of units for y_smoother birdectional RNNs, "
                                                      "int seperated by comma")
flags.DEFINE_string("X0_smoother_Dhs", X0_smoother_Dhs, "number of units for X0_smoother birdectional RNNs, "
                                                        "int seperated by comma")
flags.DEFINE_boolean("X0_use_separate_RNN", X0_use_separate_RNN, "whether use a separate RNN for getting X0")
flags.DEFINE_boolean("use_stack_rnn", use_stack_rnn, "whether use tf.contrib.rnn.stack_bidirectional_dynamic_rnn "
                                                     "or tf.nn.bidirectional_dynamic_rnn")
# ------------------------ State Space Model ------------------------- #

flags.DEFINE_boolean("dirichlet_emission", dirichlet_emission, "whether emission uses Dirichlet_ distribution")
flags.DEFINE_boolean("use_bootstrap", use_bootstrap, "whether q1 and f share the same network, "
                                                     "(ATTENTION: even if use_2_q == True, "
                                                     "f and q1 can still use different networks)")
flags.DEFINE_boolean("q_uses_true_X", q_uses_true_X, "whether q1 uses true hidden states to sample")
flags.DEFINE_boolean("use_2_q", use_2_q, "whether q uses two networks q1(x_t|x_t-1) and q2(x_t|y_t), "
                                         "if True, q_uses_true_X will be overwritten as False")
flags.DEFINE_boolean("poisson_emission", poisson_emission, "whether emission uses Poisson distribution")

# ------------------------- Inference Schemes ------------------------ #

flags.DEFINE_boolean("PSVO", PSVO, "Particle Smoothing Variational Objective (use Forward Filtering Backward Simulation)")
flags.DEFINE_boolean("SVO", SVO, "Smoothing Variational Objective (use proposal based on bRNN)")
flags.DEFINE_boolean("AESMC", AESMC, "Auto-Encoding Sequential Monte Carlo")
flags.DEFINE_boolean("IWAE", IWAE, "Importance Weighted Auto-Encoder")

flags.DEFINE_boolean("BSim_use_single_RNN", BSim_use_single_RNN, "whether Backward Simulation proposal "
                                                                 "use unidirectional RNN or bidirectional RNN")

flags.DEFINE_integer("batch_size", batch_size, "batch size")

tf.app.flags.DEFINE_string('f', '', 'kernel')

FLAGS = flags.FLAGS

model = SSM(FLAGS)

T = 2

x0s = tf.random.normal((1,2)) * 5
inputs = tf.random.normal((1,2))


sample_zs = []
sample_xs = []
for i in range(1):
    print(i)
    sample_z, sample_x = model.sample(T, x0=x0s[i:i+1])
    sample_zs.append(sample_z)
    sample_xs.append(sample_x)

sample_zs_t = tf.concat(sample_zs, axis=0)
sample_xs_t = tf.concat(sample_xs, axis=0)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sample_zs_val, sample_xs_val = sess.run([sample_zs_t, sample_xs_t])

