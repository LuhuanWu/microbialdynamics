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
Dv = 0                 # dimension of inputs. for microbio data, Dv = 15
Dev = 0                 # dimension of inputs.
n_particles = 16        # number of particles
n_bw_particles = 16  # number of subparticles sampled when augmenting the trajectory backwards
batch_size = 1          # batch size
lr = 1e-3               # learning rate
Adam_beta1 = 0.9
epochs = [600, 400, 400, 400]  # 500*100 #100*200
seed = 0

# ------------------------------- Data ------------------------------- #

# see options: utils/available_data.py
data_type = "clv_count_Dx10_obs10_noinput"
interpolation_type = 'none'  # choose from 'linear_lar', 'gp_lar', 'clv' and 'none'
interpolation_data_type = 'count_clv'

# choose samples from the data set for training. -1 indicates use default training set
training_sample_idx = [-1]
# choose samples from the test set for test. -1 indicates default test set
test_sample_idx = [-1]

# ------------------------ Networks parameters ----------------------- #
# Feed-Forward Networks (FFN), number of units in each hidden layer
# For example, [64, 64] means 2 hidden layers, 64 units in each hidden layer
q0_layers = [32]        # q(x_1|y_1) or q(x_1|y_1:T)
q1_layers = [32]        # q(x_t|x_{t-1}), including backward evolution term q(x_{t-1}|x_t)
q2_layers = [32]        # q(x_t|y_t) or q(x_t|y_1:T)
f_layers = [32]         # target evolution
h_layers = [32]         # target emission (middle step)
g_layers = [32]         # target emission

# Covariance Terms
q0_sigma_init, q0_sigma_min = 5, 1e-8
q1_sigma_init, q1_sigma_min = 5, 1e-8
q2_sigma_init, q2_sigma_min = 5, 1e-8
f_sigma_init, f_sigma_min = 5, 1e-8
g_sigma_init, g_sigma_min = 5, 1e-8

qh_sigma_init, qh_sigma_min = 5, 1e-8
h_sigma_init, h_sigma_min = 5, 1e-8

# bidirectional RNN, number of units in each LSTM cells
# For example, [32, 32] means a bRNN composed of 2 LSTM cells, 32 units in each cell
y_smoother_Dhs = [32]
X0_smoother_Dhs = [32]

f_use_residual = True

# whether use tf.contrib.rnn.stack_bidirectional_dynamic_rnn or tf.nn.bidirectional_dynamic_rnn
# check https://stackoverflow.com/a/50552539 for differences between them
use_stack_rnn = True

# ------------------------ State Space Model ------------------------- #
use_mask = True  # whether to use mask in log_ZSMC. note that mask will always be used in R_square
use_mask_interpolation = False  # whether to use mask in log_ZSMC. note that mask will always be used in R_square

f_tran_type = "MLP"          # choose from MLP, linear, clv
g_tran_type = "LDA"          # choose from MLP, LDA
g_dist_type = "multinomial"  # choose from dirichlet, poisson, multinomial and mvn

emission_use_auxiliary = True

# ------------------------- Inference Schemes ------------------------ #
# Choose one of the following objectives
PSVO = False      # Particle Smoothing Variational Objective (use Forward Filtering Backward Simulation)
SVO = True      # Smoothing Variational Objective (use proposal based on bRNN)
AESMC = False    # Auto-Encoding Sequential Monte Carlo
IWAE = False     # Importance Weighted Auto-Encoder

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
update_interp_while_train = False
update_interp_interval = 1  # 100 epochs

# --------------------- printing, data saving and evaluation params --------------------- #
# frequency to evaluate testing loss & other metrics and save results
print_freq = 10 # 100

# whether to save following into epoch folder
save_trajectory = True
save_y_hat_train = True
save_y_hat_test = True

# dir to save all results
rslt_dir_name = "test_k2"

# number of steps to predict y-hat and calculate R_square
MSE_steps = 5

# number of testing data used to save hidden trajectories, y-hat, gradient and etc
# will be clipped by number of testing data
saving_train_num = 20
saving_test_num = 20

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
flags.DEFINE_float("Adam_beta1", Adam_beta1, "Adam optimizer beta1")
flags.DEFINE_string("epochs", epochs, "list of number of epochs")

flags.DEFINE_integer("seed", seed, "random seed for np.random and tf")


# ------------------------------- Data ------------------------------- #

flags.DEFINE_string("data_type", data_type, "The type of data, chosen from toy, percentage and count.")
flags.DEFINE_string("interpolation_type", interpolation_type, "The type of interpolation, "
                                                              "chhoose from 'linear_lar', 'gp_lar', 'clv', and None")
flags.DEFINE_string("interpolation_data_type", interpolation_data_type, "The file for data interpolation")

flags.DEFINE_string("training_sample_idx", training_sample_idx, "choose samples from the dataset for training")
flags.DEFINE_string("test_sample_idx", test_sample_idx, "choose samples from the dataset for test")

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

flags.DEFINE_float("qh_sigma_init",  qh_sigma_init,  "initial value of qh_sigma")
flags.DEFINE_float("h_sigma_init",  h_sigma_init,  "initial value of h_sigma")

flags.DEFINE_float("qh_sigma_min",  qh_sigma_min,  "minimal value of qh_sigma")
flags.DEFINE_float("h_sigma_min",  h_sigma_min,  "minimal value of h_sigma")

# bidirectional RNN
flags.DEFINE_string("y_smoother_Dhs", y_smoother_Dhs, "number of units for y_smoother birdectional RNNs, "
                                                      "int seperated by comma")
flags.DEFINE_string("X0_smoother_Dhs", X0_smoother_Dhs, "number of units for X0_smoother birdectional RNNs, "
                                                        "int seperated by comma")

flags.DEFINE_boolean("f_use_residual", f_use_residual, "whether use batch normalization and residual for transition")
flags.DEFINE_boolean("use_stack_rnn", use_stack_rnn, "whether use tf.contrib.rnn.stack_bidirectional_dynamic_rnn "
                                                     "or tf.nn.bidirectional_dynamic_rnn")
# ------------------------ State Space Model ------------------------- #
flags.DEFINE_boolean("use_mask", use_mask, "whether to use mask for missing observations")
flags.DEFINE_boolean("use_mask_interpolation", use_mask_interpolation, "whether to use mask interpolation for missing observations")

flags.DEFINE_string("f_tran_type", f_tran_type, "type of f transformation, choose from MLP, linear, clv and clv_original")
flags.DEFINE_string("g_tran_type", g_tran_type, "type of g transformation, choose from MLP and LDA")
flags.DEFINE_string("g_dist_type", g_dist_type, "type of g distribution, chosen from dirichlet, poisson, mvn and multinomial")

flags.DEFINE_boolean("emission_use_auxiliary", emission_use_auxiliary, "whether to use auxiliary variables in emission")

# ------------------------- Inference Schemes ------------------------ #

flags.DEFINE_boolean("PSVO", PSVO, "Particle Smoothing Variational Objective (use Forward Filtering Backward Simulation)")
flags.DEFINE_boolean("SVO", SVO, "Smoothing Variational Objective (use proposal based on bRNN)")
flags.DEFINE_boolean("AESMC", AESMC, "Auto-Encoding Sequential Monte Carlo")
flags.DEFINE_boolean("IWAE", IWAE, "Importance Weighted Auto-Encoder")

flags.DEFINE_integer("n_bw_particles", n_bw_particles, "number of particles used for"
                                                                                     " each trajectory in "
                                                                                     "backward simulation proposal")

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

FLAGS = flags.FLAGS

if __name__ == "__main__":
    tf.app.run()
