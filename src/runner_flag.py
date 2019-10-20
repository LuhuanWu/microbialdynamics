import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from src.runner import main

np.warnings.filterwarnings('ignore')          # to avoid np deprecation warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # to avoid lots of log about the device
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'   # hack to avoid OS bug...

print("the code is written in:")
print("\t tensorflow version: 1.12.0")
print("\t tensorflow_probability version: 0.5.0")

print("the system uses:")
print("\t tensorflow version:", tf.__version__)
print("\t tensorflow_probability version:", tfp.__version__)


# --------------------- Training Hyperparameters --------------------- #
Dx = 10                # dimension of hidden states
Dy = 11                  # dimension of observations. for microbio data, Dy = 11
Dv = 16                 # dimension of inputs. for microbio data, Dv = 15
Dev = 10                 # dimension of inputs.
n_particles = 32        # number of particles
batch_size = 1          # batch size
lr = 1e-3               # learning rate
epochs = [50]  # 500*100 #100*200
seed = 0

# ------------------------------- Data ------------------------------- #
# True: generate data set from simulation
# False: read data set from the file
generate_training_data = False

# see options: utils/available_data.py
data_type = "count"
interpolation_type = 'count_clv' #'count_clv'  # choose from 'linear_lar', 'gp_lar', 'count_clv' and 'none'

# choose samples from the data set for training. -1 indicates use default training set
training_sample_idx = [-1]
# choose samples from the test set for test. -1 indicates default test set
test_sample_idx = [-1]

isPython2 = False

# time, n_train and n_test will be overwritten if loading data from the file
time = 5
n_train = 2 * batch_size
n_test = 2 * batch_size

# ------------------------ Networks parameters ----------------------- #
# Feed-Forward Networks (FFN), number of units in each hidden layer
# For example, [64, 64] means 2 hidden layers, 64 units in each hidden layer
q0_layers = [16]        # q(x_1|y_1) or q(x_1|y_1:T)
q1_layers = [16]        # q(x_t|x_{t-1}), including backward evolution term q(x_{t-1}|x_t)
q2_layers = [16]        # q(x_t|y_t) or q(x_t|y_1:T)
f_layers = [16]         # target evolution
h_layers = [16]         # target emission (middle step)
g_layers = [16]         # target emission

# number of f^power
f_power = 1

# Covariance Terms
q0_sigma_init, q0_sigma_min = 5, 1e-8
q1_sigma_init, q1_sigma_min = 5, 1e-8
q2_sigma_init, q2_sigma_min = 5, 1e-8
f_sigma_init, f_sigma_min = 5, 1e-8
h_sigma_init, h_sigma_min = 5, 1e-8
g_sigma_init, g_sigma_min = 5, 1e-8

# if q, f and g networks also output covariance (sigma)
output_cov = False

# whether the networks only output diagonal value of cov matrix
diag_cov = False

# bidirectional RNN, number of units in each LSTM cells
# For example, [32, 32] means a bRNN composed of 2 LSTM cells, 32 units in each cell
y_smoother_Dhs = [16]
X0_smoother_Dhs = [16]

# whether use a separate RNN for getting X0
X0_use_separate_RNN = True

# whether use tf.contrib.rnn.stack_bidirectional_dynamic_rnn or tf.nn.bidirectional_dynamic_rnn
# check https://stackoverflow.com/a/50552539 for differences between them
use_stack_rnn = True

# ------------------------ State Space Model ------------------------- #
use_mask = False  # whether to use mask in log_ZSMC. note that mask will always be used in R_square

# whether emission uses Dirichlet distribution
emission = "multinomial"  # choose from dirichlet, poisson, multinomial and mvn

# whether use two step emission
two_step_emission = True
two_step_emission_type = "inv_lar"  # choose from MLP and inv_lar

# f_transformation
f_transformation = "MLP"  # choose from MLP, linear, clv, clv_original

# whether q1 (evolution term in proposal) and f share the same network
# (ATTENTION: even if use_2_q == True, f and q1 can still use different networks)
use_bootstrap = True

# should q use true_X to sample? (useful for debugging)
q_uses_true_X = False

# if q uses two networks q1(x_t|x_t-1) and q2(x_t|y_t)
# if True, q_uses_true_X will be overwritten as False
use_2_q = True

log_dynamics = False  # whether to set latent dynamics in the log space
lar_dynamics = False  # log additive ratio transformation
f_final_scaling = 3   # when use log_dynamics/lar_dynamics, activation
                      # for f's final layer is tanh, choose a scaling for f

# ------------------------- Inference Schemes ------------------------ #
# Choose one of the following objectives
PSVO = False      # Particle Smoothing Variational Objective (use Forward Filtering Backward Simulation)
SVO = True      # Smoothing Variational Objective (use proposal based on bRNN)
AESMC = False    # Auto-Encoding Sequential Monte Carlo
IWAE = False     # Importance Weighted Auto-Encoder

# number of subparticles sampled when augmenting the trajectory backwards
n_particles_for_BSim_proposal = 16

# whether Backward Simulation proposal use unidirectional RNN or bidirectional RNN
BSim_use_single_RNN = False

# ----------------------------- Training ----------------------------- #

# stop training early if validation set does not improve
early_stop_patience = 400

# reduce learning rate when testing loss doesn't improve for some time
lr_reduce_patience = 50

# the factor to reduce lr, new_lr = old_lr * lr_reduce_factor
lr_reduce_factor = 1 / np.sqrt(2)

# minimum lr
min_lr = lr / 100

# some interpolation and learning schemes
update_interp_while_train = True
update_interp_interval = 1  # 100 epochs

# --------------------- printing, data saving and evaluation params --------------------- #
# frequency to evaluate testing loss & other metrics and save results
print_freq = 2 # 100

# whether to save following into epoch folder
save_trajectory = False
save_y_hat_train = False
save_y_hat_test = False

# dir to save all results
rslt_dir_name = "test_interpolate"

# number of steps to predict y-hat and calculate R_square
MSE_steps = 5

# number of testing data used to save hidden trajectories, y-hat, gradient and etc
# will be clipped by number of testing data
saving_train_num = 5
saving_test_num = 5

# whether to save tensorboard
save_tensorboard = False

# whether to save model
save_model = False

epochs = ",".join([str(epoch) for epoch in epochs])

q0_layers = ",".join([str(x) for x in q0_layers])
q1_layers = ",".join([str(x) for x in q1_layers])
q2_layers = ",".join([str(x) for x in q2_layers])
f_layers = ",".join([str(x) for x in f_layers])
h_layers = ",".join([str(x) for x in h_layers])
g_layers = ",".join([str(x) for x in g_layers])
y_smoother_Dhs = ",".join([str(x) for x in y_smoother_Dhs])
X0_smoother_Dhs = ",".join([str(x) for x in X0_smoother_Dhs])

training_sample_idx = ",".join([str(x) for x in training_sample_idx])
test_sample_idx = ",".join([str(x) for x in test_sample_idx])


# ================================================ tf.flags ================================================ #

flags = tf.app.flags


# --------------------- Training Hyperparameters --------------------- #

flags.DEFINE_integer("Dx", Dx, "dimension of hidden states")
flags.DEFINE_integer("Dy", Dy, "dimension of observations")
flags.DEFINE_integer("Dv", Dv, "dimension of inputs")
flags.DEFINE_integer("Dev", Dev, "input embedding size")

flags.DEFINE_integer("n_particles", n_particles, "number of particles")
flags.DEFINE_integer("batch_size", batch_size, "batch size")
flags.DEFINE_float("lr", lr, "learning rate")
flags.DEFINE_string("epochs", epochs, "list of number of epochs")

flags.DEFINE_integer("seed", seed, "random seed for np.random and tf")


# ------------------------------- Data ------------------------------- #

flags.DEFINE_boolean("generate_training_data", generate_training_data, "True: generate data set from simulation; "
                                                                   "False: read data set from the file")
flags.DEFINE_string("data_type", data_type, "The type of data, chosen from toy, percentage and count.")
flags.DEFINE_string("interpolation_type", interpolation_type, "The type of interpolation, "
                                                              "chhoose from 'linear_lar', 'gp_lar', 'clv', and None")

flags.DEFINE_string("training_sample_idx", training_sample_idx, "choose samples from the dataset for training")
flags.DEFINE_string("test_sample_idx", test_sample_idx, "choose samples from the dataset for test")

flags.DEFINE_boolean("isPython2", isPython2, "Was the data pickled in python 2?")

flags.DEFINE_integer("time", time, "number of timesteps for simulated data")
flags.DEFINE_integer("n_train", n_train, "number of trajactories for traning set")
flags.DEFINE_integer("n_test", n_test, "number of trajactories for testing set")

# ------------------------ Networks parameters ----------------------- #
# Feed-Forward Network (FFN) architectures
flags.DEFINE_string("q0_layers", q0_layers, "architecture for q0 network, int seperated by comma, "
                                            "for example: '50,50' ")
flags.DEFINE_string("q1_layers", q1_layers, "architecture for q1 network, int seperated by comma, "
                                            "for example: '50,50' ")
flags.DEFINE_string("q2_layers", q2_layers, "architecture for q2 network, int seperated by comma, "
                                            "for example: '50,50' ")
flags.DEFINE_string("f_layers",  f_layers,  "architecture for f network, int seperated by comma, "
                                            "for example: '50,50' ")
flags.DEFINE_string("h_layers",  h_layers,  "architecture for h network, int seperated by comma, "
                                            "for example: '50,50' ")
flags.DEFINE_string("g_layers",  g_layers,  "architecture for g network, int seperated by comma, "
                                            "for example: '50,50' ")

flags.DEFINE_integer("f_power", f_power, "number of f^power")

flags.DEFINE_float("q0_sigma_init", q0_sigma_init, "initial value of q0_sigma")
flags.DEFINE_float("q1_sigma_init", q1_sigma_init, "initial value of q1_sigma")
flags.DEFINE_float("q2_sigma_init", q2_sigma_init, "initial value of q2_sigma")
flags.DEFINE_float("f_sigma_init",  f_sigma_init,  "initial value of f_sigma")
flags.DEFINE_float("h_sigma_init",  h_sigma_init,  "initial value of h_sigma")
flags.DEFINE_float("g_sigma_init",  g_sigma_init,  "initial value of g_sigma")

flags.DEFINE_float("q0_sigma_min", q0_sigma_min, "minimal value of q0_sigma")
flags.DEFINE_float("q1_sigma_min", q1_sigma_min, "minimal value of q1_sigma")
flags.DEFINE_float("q2_sigma_min", q2_sigma_min, "minimal value of q2_sigma")
flags.DEFINE_float("f_sigma_min",  f_sigma_min,  "minimal value of f_sigma")
flags.DEFINE_float("h_sigma_min",  h_sigma_min,  "minimal value of h_sigma")
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
flags.DEFINE_boolean("use_mask", use_mask, "whether to use mask for missing observations")
flags.DEFINE_string("emission", emission, "type of emission, chosen from dirichlet, poisson and mvn")
flags.DEFINE_boolean("two_step_emission", two_step_emission, "whether add a Gaussian layer in the middle of emission")
flags.DEFINE_string("two_step_emission_type", two_step_emission_type, "choose from inv_lar and MLP")

flags.DEFINE_string("f_transformation", f_transformation, "type of f_transformation, choose from MLP, linear, clv and clv_original")

flags.DEFINE_boolean("use_bootstrap", use_bootstrap, "whether q1 and f share the same network, "
                                                     "(ATTENTION: even if use_2_q == True, "
                                                     "f and q1 can still use different networks)")
flags.DEFINE_boolean("q_uses_true_X", q_uses_true_X, "whether q1 uses true hidden states to sample")
flags.DEFINE_boolean("use_2_q", use_2_q, "whether q uses two networks q1(x_t|x_t-1) and q2(x_t|y_t), "
                                         "if True, q_uses_true_X will be overwritten as False")
flags.DEFINE_boolean("log_dynamics", log_dynamics, "whether the dynamics happen in log space")
flags.DEFINE_boolean("lar_dynamics", lar_dynamics, "whether the dynamics happen in ldr space")
flags.DEFINE_float("f_final_scaling", f_final_scaling, "scaling of the final layer of transition MLP")

# ------------------------- Inference Schemes ------------------------ #

flags.DEFINE_boolean("PSVO", PSVO, "Particle Smoothing Variational Objective (use Forward Filtering Backward Simulation)")
flags.DEFINE_boolean("SVO", SVO, "Smoothing Variational Objective (use proposal based on bRNN)")
flags.DEFINE_boolean("AESMC", AESMC, "Auto-Encoding Sequential Monte Carlo")
flags.DEFINE_boolean("IWAE", IWAE, "Importance Weighted Auto-Encoder")

flags.DEFINE_integer("n_particles_for_BSim_proposal", n_particles_for_BSim_proposal, "number of particles used for"
                                                                                     " each trajectory in "
                                                                                     "backward simulation proposal")
flags.DEFINE_boolean("BSim_use_single_RNN", BSim_use_single_RNN, "whether Backward Simulation proposal "
                                                                 "use unidirectional RNN or bidirectional RNN")

# ----------------------------- Training ----------------------------- #

flags.DEFINE_integer("early_stop_patience", early_stop_patience,
                     "stop training early if validation set does not improve for certain epochs")

flags.DEFINE_integer("lr_reduce_patience", lr_reduce_patience,
                     "educe learning rate when testing loss doesn't improve for some time")
flags.DEFINE_float("lr_reduce_factor", lr_reduce_factor,
                   "the factor to reduce learning rate, new_lr = old_lr * lr_reduce_factor")
flags.DEFINE_float("min_lr", min_lr, "minimum learning rate")

flags.DEFINE_boolean("update_interp_while_train", update_interp_while_train,
                     "whether to update the interpolation data while training")
flags.DEFINE_integer("update_interp_interval", update_interp_interval, "the interval (number of epochs) of updating "
                                                                       "the interpolation")

# --------------------- printing, data saving and evaluation params --------------------- #

flags.DEFINE_integer("print_freq", print_freq, "frequency to evaluate testing loss & other metrics and save results")

flags.DEFINE_boolean("save_trajectory", save_trajectory, "whether to save hidden trajectories during training")
flags.DEFINE_boolean("save_y_hat_train", save_y_hat_train, "whether to save k-step y-hat-train during training")
flags.DEFINE_boolean("save_y_hat_test", save_y_hat_test, "whether to save k-step y-hat-test during training")

flags.DEFINE_string("rslt_dir_name", rslt_dir_name, "dir to save all results")
flags.DEFINE_integer("MSE_steps", MSE_steps, "number of steps to predict y-hat and calculate R_square")

flags.DEFINE_integer("saving_train_num", saving_test_num, "number of training data used to "
                                               "save hidden trajectories, y-hat, gradient and etc, "
                                               "will be clipped by number of testing data")

flags.DEFINE_integer("saving_test_num", saving_test_num, "number of testing data used to "
                                               "save hidden trajectories, y-hat, gradient and etc, "
                                               "will be clipped by number of testing data")

flags.DEFINE_boolean("save_tensorboard", save_tensorboard, "whether to save tensorboard")
flags.DEFINE_boolean("save_model", save_model, "whether to save model")

# for debug purpose
flags.DEFINE_boolean("print_f", False, "whether to print f or not")
flags.DEFINE_integer("print_f_frequency", 2, "frequency of printing f")

FLAGS = flags.FLAGS

if __name__ == "__main__":
    tf.app.run()
