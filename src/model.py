import tensorflow as tf
from tensorflow.keras.layers import Dense

from src.transformation.MLP import MLP_transformation
from src.transformation.linear import tf_linear_transformation
from src.transformation.LDA import LDA_transformation
from src.transformation.clv import clv_transformation
from src.transformation.identity import identity_transformation
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

    def __init__(self, FLAGS):
        assert FLAGS.g_dist_type in SUPPORTED_EMISSION.keys(), "g_dist_type must be one of " + str(SUPPORTED_EMISSION.keys())

        self.Dx = FLAGS.Dx
        self.Dy = FLAGS.Dy
        self.Dv = FLAGS.Dv  # dimension of the input. 0 indicates not using input
        self.Dev = FLAGS.Dev

        self.batch_size = FLAGS.batch_size


        # Feed-Forward Network (FFN) architectures
        self.q0_layers = [int(x) for x in FLAGS.q0_layers.split(",")]
        self.q1_layers = [int(x) for x in FLAGS.q1_layers.split(",")]
        self.q2_layers = [int(x) for x in FLAGS.q2_layers.split(",")]
        self.f_layers  = [int(x) for x in FLAGS.f_layers.split(",")]
        self.h_layers  = [int(x) for x in FLAGS.h_layers.split(",")]
        self.g_layers  = [int(x) for x in FLAGS.g_layers.split(",")]

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

        self.f_use_residual              = FLAGS.f_use_residual
        self.use_stack_rnn             = FLAGS.use_stack_rnn

        self.PSVO                      = FLAGS.PSVO
        self.SVO                       = FLAGS.SVO

        self.init_placeholder()
        self.init_trans()
        self.init_dist()
        self.init_RNNs()
        self.init_input_embedding()

    def init_placeholder(self):
        self.obs = tf.placeholder(tf.float32, shape=(self.batch_size, None, self.Dy), name="obs")
        self.input = tf.placeholder(tf.float32, shape=(self.batch_size, None, self.Dv), name="input")
        self.time = tf.placeholder(tf.int32, shape=(), name="time")
        self.mask = tf.placeholder(tf.bool, shape=(self.batch_size, None), name="mask")
        self.mask_weight = tf.placeholder(tf.float32, shape=(), name="mask_weight")
        self.time_interval = tf.placeholder(tf.float32, shape=(self.batch_size, None), name="time_interval")
        self.extra_inputs = tf.placeholder(tf.float32, shape=(self.batch_size, None), name="extra_inputs")
        self.training = tf.placeholder(tf.bool, shape=(), name="training")

    def init_trans(self):
        if self.f_tran_type == "MLP":
            self.f_tran = MLP_transformation(self.f_layers, self.Dx,
                                             use_residual=self.f_use_residual, training=self.training, name="f_tran")
        elif self.f_tran_type == "linear":
            self.f_tran = tf_linear_transformation(self.Dx, self.Dev)
        elif self.f_tran_type == "clv":
            self.f_tran = clv_transformation(self.Dx, self.Dev)
        else:
            raise ValueError("Invalid value for f transformation. Must choose from MLP, linear and clv.")

        if self.g_tran_type == "MLP":
            self.g_tran = MLP_transformation(self.g_layers, self.Dy, name="g_tran")
        elif self.g_tran_type == "LDA":
            self.g_tran = LDA_transformation(self.Dx, self.Dy, training=self.training)
        else:
            raise ValueError("Invalid value for g transformation. Must choose from MLP and LDA.")

        self.q0_tran = MLP_transformation(self.q0_layers, self.Dx, name="q0_tran")

        if self.use_2_q:
            self.q2_tran = MLP_transformation(self.q2_layers, self.Dx, name="q2_tran")
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

    def init_input_embedding(self):
        self.input_embedding_layer = Dense(self.Dev,
                                           activation="linear",
                                           kernel_initializer="he_uniform",
                                           name="input_embedding")
        self.input_embedding = self.input_embedding_layer(self.input)

    def sample(self, T, x0=None, x0_mu=None, inputs=None):
        """
        Sampling using f and g
        :param T: the length of sample sequence
        :param x0_mu: transformation input for initial distribution, (1, Dy)
        :param x0: initial hidden state, (1, Dx)
        :param inputs: (T, Dv)
        :return: (x_{0:T-1), y_{0:T-1}), a tuple of two tensors
        """

        if x0 is None:
            assert x0_mu is not None
            # TODO: fix this
            assert x0_mu.shape == (1, self.Dy)
            x0 = self.q0_dist.sample(x0_mu)

        else:
            assert x0.shape == (1, self.Dx)

        if inputs is not None:
            batch_size, T_inputs, Dv = inputs.shape
            assert Dv == self.Dv
            assert batch_size == 1
            assert T_inputs == T

            inputs_embedding = self.input_embedding_layer(inputs)

            assert inputs_embedding.shape == (1, T, self.Dev)

        # TODO: fix poisson sampling
        y0 = self.g_dist.sample(x0)

        x_list = [x0]
        y_list = [y0]
        for t in range(1, T):
            if inputs is None:
                xt = self.f_dist.sample(x_list[-1])
            else:
                xt = self.f_dist.sample(tf.concat((x_list[-1], inputs_embedding[:,t-1]), axis=-1))
            yt = self.g_dist.sample(xt)

            x_list.append(xt)
            y_list.append(yt)

        xs = tf.stack(x_list, axis=0)  # (T, n_batches, Dx)
        ys = tf.stack(y_list, axis=0)  # (T, n_batches, Dy)

        assert xs.shape == (T, 1, self.Dx)
        assert ys.shape == (T, 1, self.Dy)

        xs = tf.transpose(xs, (1, 0, 2))  # (n_batches, T, Dx)
        ys = tf.transpose(ys, (1, 0, 2))  # (n_batches, T, Dy)s
        return xs, ys


