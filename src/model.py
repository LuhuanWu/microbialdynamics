import tensorflow as tf
from tensorflow.keras.layers import Dense

from src.transformation.MLP import MLP_transformation
from src.transformation.linear import tf_linear_transformation
from src.transformation.LDA import LDA_transformation
from src.transformation.clv import clv_transformation
from src.transformation.ilr_clv import ilr_clv_transformation
from src.transformation.identity import identity_transformation
from src.transformation.inv_alr import inv_alr_transformation
from src.transformation.inv_ilr import inv_ilr_transformation
from src.distribution.mvn import tf_mvn
from src.distribution.poisson import tf_poisson
from src.distribution.dirichlet import tf_dirichlet
from src.distribution.multinomial import tf_multinomial

SUPPORTED_EMISSION = dict(dirichlet=tf_dirichlet,
                          poisson=tf_poisson,
                          mvn=tf_mvn,
                          multinomial=tf_multinomial,
                          LDA=LDA_transformation)


class SSM(object):
    """
    state space model
    keeps all placeholders
    keeps q, f, g and obs smoother
    """

    def __init__(self, FLAGS, theta):
        assert FLAGS.g_dist_type in SUPPORTED_EMISSION.keys(), "g_dist_type must be one of " +\
                                                               str(SUPPORTED_EMISSION.keys())

        self.Dx = FLAGS.Dx
        self.Dy = FLAGS.Dy
        self.Dv = FLAGS.Dv  # dimension of the input. 0 indicates not using input

        self.theta = theta
        self.exist_in_group_dynamics = FLAGS.exist_in_group_dynamics
        self.use_L0 = FLAGS.use_L0
        self.inference_schedule = FLAGS.inference_schedule
        self.reg_coef = FLAGS.reg_coef

        self.batch_size = FLAGS.batch_size

        # Feed-Forward Network (FFN) architectures
        self.q0_layers = [int(x) for x in FLAGS.q0_layers.split(",") if x != '']
        self.q1_layers = [int(x) for x in FLAGS.q1_layers.split(",") if x != '']
        self.q2_layers = [int(x) for x in FLAGS.q2_layers.split(",") if x != '']
        self.f_layers  = [int(x) for x in FLAGS.f_layers.split(",") if x != '']
        self.h_layers  = [int(x) for x in FLAGS.h_layers.split(",") if x != '']
        self.g_layers  = [int(x) for x in FLAGS.g_layers.split(",") if x != '']

        self.q0_sigma_init, self.q0_sigma_min = FLAGS.q0_sigma_init, FLAGS.q0_sigma_min
        self.q1_sigma_init, self.q1_sigma_min = FLAGS.q1_sigma_init, FLAGS.q1_sigma_min
        self.q2_sigma_init, self.q2_sigma_min = FLAGS.q2_sigma_init, FLAGS.q2_sigma_min
        self.f_sigma_init,  self.f_sigma_min  = FLAGS.f_sigma_init, FLAGS.f_sigma_min
        self.g_sigma_init,  self.g_sigma_min  = FLAGS.f_sigma_init, FLAGS.g_sigma_min

        self.qh_sigma_init, self.qh_sigma_min = FLAGS.qh_sigma_init, FLAGS.qh_sigma_min
        self.h_sigma_init,  self.h_sigma_min  = FLAGS.h_sigma_init, FLAGS.h_sigma_min

        # bidirectional RNN architectures
        self.y_smoother_Dhs  = [int(x) for x in FLAGS.y_smoother_Dhs.split(",")]
        self.X0_smoother_Dhs = [int(x) for x in FLAGS.X0_smoother_Dhs.split(",")]

        self.use_bootstrap             = True
        self.use_2_q                   = True
        self.f_tran_type               = FLAGS.f_tran_type
        self.g_tran_type               = FLAGS.g_tran_type
        self.g_dist_type               = FLAGS.g_dist_type

        self.f_use_residual            = FLAGS.f_use_residual
        self.use_stack_rnn             = FLAGS.use_stack_rnn

        self.PSVO                      = FLAGS.PSVO
        self.SVO                       = FLAGS.SVO

        self.init_placeholder()
        self.init_trans()
        self.init_dist()
        self.init_RNNs()

    def init_placeholder(self):
        self.obs = tf.placeholder(tf.float32, shape=(self.batch_size, None, self.Dy), name="obs")
        self.input = tf.placeholder(tf.float32, shape=(self.batch_size, None, self.Dv), name="input")
        self.time = tf.placeholder(tf.int32, shape=(), name="time")
        self.mask = tf.placeholder(tf.bool, shape=(self.batch_size, None), name="mask")
        self.mask_weight = tf.placeholder(tf.float32, shape=(), name="mask_weight")
        self.time_interval = tf.placeholder(tf.float32, shape=(self.batch_size, None), name="time_interval")
        self.training = tf.placeholder(tf.bool, shape=(), name="training")
        self.annealing_frac = tf.placeholder(tf.float32, shape=(), name="annealing_frac")

    def init_trans(self):
        if self.f_tran_type == "MLP":
            self.f_tran = MLP_transformation(self.f_layers, self.Dx,
                                             use_residual=self.f_use_residual, training=self.training, name="f_tran")
        elif self.f_tran_type == "linear":
            self.f_tran = tf_linear_transformation(self.Dx, self.Dv)
        elif self.f_tran_type == "clv":
            assert self.Dx == self.Dy - 1
            self.f_tran = clv_transformation(self.Dx, self.Dv,
                                             reg_coef=self.reg_coef, annealing_frac=self.annealing_frac)
        elif self.f_tran_type == "ilr_clv":
            assert self.Dx == self.Dy - 1
            assert self.theta.shape == (self.Dy - 1, self.Dy)
            self.f_tran = ilr_clv_transformation(self.theta, self.Dv, self.exist_in_group_dynamics,
                                                 use_L0=self.use_L0, inference_schedule=self.inference_schedule,
                                                 training=self.training, annealing_frac=self.annealing_frac)
        else:
            raise ValueError("Invalid value for f transformation. Must choose from MLP, linear and clv.")

        if self.g_tran_type == "MLP":
            self.g_tran = MLP_transformation(self.g_layers, self.Dy, name="g_tran")
        elif self.g_tran_type == "LDA":
            self.g_tran = LDA_transformation(self.Dx, self.Dy)
        elif self.g_tran_type == "inv_alr":
            self.g_tran = inv_alr_transformation()
        elif self.g_tran_type == "inv_ilr":
            self.g_tran = inv_ilr_transformation(self.theta)
        else:
            raise ValueError("Invalid value for g transformation. Must choose from MLP and LDA.")

        self.q0_tran = MLP_transformation(self.q0_layers, self.Dx,
                                          initialize_around_zero=self.f_tran_type == "clv",
                                          name="q0_tran")

        if self.use_2_q:
            self.q2_tran = MLP_transformation(self.q2_layers, self.Dx,
                                              initialize_around_zero=self.f_tran_type == "clv",
                                              name="q2_tran")
        else:
            self.q2_tran = None

        if self.PSVO:
            self.BSim_q_init_tran = MLP_transformation(self.q0_layers, self.Dx, name="BSim_q_init_tran")
            self.q1_inv_tran = MLP_transformation(self.q1_layers, self.Dx,
                                                  use_residual=self.f_use_residual, training=self.training,
                                                  name="q1_inv_tran")
            self.BSim_q2_tran = MLP_transformation(self.q2_layers, self.Dx, name="BSim_q2_tran")

        if self.use_bootstrap:
            self.q1_tran = self.f_tran
        else:
            self.q1_tran = MLP_transformation(self.q1_layers, self.Dx,
                                              use_residual=self.f_use_residual, training=self.training,
                                              name="q1_tran")

        self.qh_tran = identity_transformation()
        self.h_tran = identity_transformation()

    def init_dist(self):
        self.q0_dist = tf_mvn(self.q0_tran,
                              sigma_init=self.q0_sigma_init,
                              sigma_min=self.q0_sigma_min,
                              name="q0_dist")

        self.q1_dist = tf_mvn(self.q1_tran,
                              sigma_init=self.q1_sigma_init,
                              sigma_min=self.q1_sigma_min,
                              name="q1_dist")
        if self.use_2_q:
            self.q2_dist = tf_mvn(self.q2_tran,
                                  sigma_init=self.q2_sigma_init,
                                  sigma_min=self.q2_sigma_min,
                                  name="q2_dist")
        else:
            self.q2_dist = None

        if self.PSVO:
            self.Bsim_q_init_dist = tf_mvn(self.BSim_q_init_tran,
                                           sigma_init=self.q0_sigma_init,
                                           sigma_min=self.q0_sigma_min,
                                           name="BSim_q_init_dist")

            self.q1_inv_dist = tf_mvn(self.q1_inv_tran,
                                      sigma_init=self.q1_sigma_init,
                                      sigma_min=self.q1_sigma_min,
                                      name="q1_inv_dist")
            self.BSim_q2_dist = tf_mvn(self.BSim_q2_tran,
                                       sigma_init=self.q2_sigma_init,
                                       sigma_min=self.q2_sigma_min,
                                       name="BSim_q2_dist")

        if self.use_bootstrap:
            self.f_dist = self.q1_dist
        else:
            self.f_dist = tf_mvn(self.f_tran,
                                 sigma_init=self.f_sigma_init,
                                 sigma_min=self.f_sigma_min,
                                 name="f_dist")

        if self.g_dist_type == "mvn":
            self.g_dist = tf_mvn(self.g_tran, name="g_dist",
                                 sigma_init=self.g_sigma_init,
                                 sigma_min=self.g_sigma_min)
        else:
            self.g_dist = SUPPORTED_EMISSION[self.g_dist_type](self.g_tran, name="g_dist")

        self.qh_dist = tf_mvn(self.qh_tran, name="qh_dist",
                              sigma_init=self.qh_sigma_init,
                              sigma_min=self.qh_sigma_min)
        self.h_dist = tf_mvn(self.h_tran, name="h_dist",
                             sigma_init=self.h_sigma_init,
                             sigma_min=self.h_sigma_min)

    def init_RNNs(self):
        if self.SVO or self.PSVO:
            y_smoother_f = [tf.contrib.rnn.LSTMBlockCell(Dh, name="y_smoother_f_{}".format(i))
                            for i, Dh in enumerate(self.y_smoother_Dhs)]
            y_smoother_b = [tf.contrib.rnn.LSTMBlockCell(Dh, name="y_smoother_b_{}".format(i))
                            for i, Dh in enumerate(self.y_smoother_Dhs)]
            if not self.use_stack_rnn:
                y_smoother_f = tf.nn.rnn_cell.MultiRNNCell(y_smoother_f)
                y_smoother_b = tf.nn.rnn_cell.MultiRNNCell(y_smoother_b)

            X0_smoother_f = [tf.contrib.rnn.LSTMBlockCell(Dh, name="X0_smoother_f_{}".format(i))
                             for i, Dh in enumerate(self.X0_smoother_Dhs)]
            X0_smoother_b = [tf.contrib.rnn.LSTMBlockCell(Dh, name="X0_smoother_b_{}".format(i))
                             for i, Dh in enumerate(self.X0_smoother_Dhs)]
            if not self.use_stack_rnn:
                X0_smoother_f = tf.nn.rnn_cell.MultiRNNCell(X0_smoother_f)
                X0_smoother_b = tf.nn.rnn_cell.MultiRNNCell(X0_smoother_b)

            self.bRNN = (y_smoother_f, y_smoother_b, X0_smoother_f, X0_smoother_b)

        else:
            self.bRNN = None
