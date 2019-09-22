import tensorflow as tf
from tensorflow.keras.layers import Dense

from src.transformation.flow import NF
from src.transformation.MLP import MLP_transformation
from src.transformation.linear import tf_linear_transformation
from src.transformation.clv import clv_transformation
from src.distribution.mvn import tf_mvn
from src.distribution.poisson import tf_poisson
from src.distribution.dirichlet import tf_dirichlet
from src.distribution.multinomial import tf_multinomial

SUPPORTED_EMISSION = dict(dirichlet=tf_dirichlet, poisson=tf_poisson, mvn=tf_mvn, multinomial=tf_multinomial)


class SSM(object):
    """
    state space model
    keeps all placeholders
    keeps q, f, g and obs smoother
    """

    def __init__(self, FLAGS):
        assert FLAGS.emission in SUPPORTED_EMISSION.keys(), "Emission must be one of " + str(SUPPORTED_EMISSION.keys())

        self.Dx = FLAGS.Dx
        self.Dy = FLAGS.Dy
        self.Dv = FLAGS.Dv  # dimension of the input. 0 indicates not using input
        self.Dev = FLAGS.Dev

        self.batch_size = FLAGS.batch_size

        self.f_transformation = FLAGS.f_transformation

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
        self.h_sigma_init,  self.h_sigma_min  = FLAGS.h_sigma_init, FLAGS.h_sigma_min
        self.g_sigma_init,  self.g_sigma_min  = FLAGS.f_sigma_init, FLAGS.g_sigma_min

        # bidirectional RNN architectures
        self.y_smoother_Dhs  = [int(x) for x in FLAGS.y_smoother_Dhs.split(",")]
        self.X0_smoother_Dhs = [int(x) for x in FLAGS.X0_smoother_Dhs.split(",")]

        self.output_cov                = FLAGS.output_cov
        self.diag_cov                  = FLAGS.diag_cov

        self.use_bootstrap             = FLAGS.use_bootstrap
        self.use_2_q                   = FLAGS.use_2_q
        self.emission                  = FLAGS.emission
        self.two_step_emission         = FLAGS.two_step_emission

        self.X0_use_separate_RNN       = FLAGS.X0_use_separate_RNN
        self.use_stack_rnn             = FLAGS.use_stack_rnn

        self.PSVO                      = FLAGS.PSVO
        self.SVO                       = FLAGS.SVO
        self.BSim_use_single_RNN       = FLAGS.BSim_use_single_RNN

        self.log_dynamics              = FLAGS.log_dynamics
        self.lar_dynamics              = FLAGS.lar_dynamics
        self.f_final_scaling           = FLAGS.f_final_scaling

        self.init_placeholder()
        self.init_trans()
        self.init_dist()
        self.init_RNNs()
        self.init_input_embedding()

    def init_placeholder(self):
        self.obs = tf.placeholder(tf.float32, shape=(self.batch_size, None, self.Dy), name="obs")
        self.hidden = tf.placeholder(tf.float32, shape=(self.batch_size, None, self.Dx), name="hidden")
        self.input = tf.placeholder(tf.float32, shape=(self.batch_size, None, self.Dv), name="input")
        self.time = tf.placeholder(tf.int32, shape=(), name="time")
        self.mask = tf.placeholder(tf.bool, shape=(self.batch_size, None), name="mask")
        self.time_interval = tf.placeholder(tf.float32, shape=(self.batch_size, None), name="time_interval")
        self.extra_inputs = tf.placeholder(tf.float32, shape=(self.batch_size, None), name="extra_inputs")

    def init_trans(self):
        if self.f_transformation == "MLP":
            if self.log_dynamics or self.lar_dynamics:
                final_activation = "tanh"
                final_scaling = self.f_final_scaling
            else:
                final_activation = "linear"
                final_scaling = 1
            self.f_tran = MLP_transformation(self.f_layers, self.Dx,
                                             output_cov=self.output_cov,
                                             diag_cov=self.diag_cov,
                                             final_activation=final_activation,
                                             final_scaling=final_scaling,
                                             name="f_tran")
        elif self.f_transformation == "linear":
            A = tf.Variable(tf.eye(self.Dx+self.Dev, self.Dx))
            b = tf.Variable(tf.zeros((self.Dx, )))
            self.f_tran = tf_linear_transformation(params=(A, b))

        elif self.f_transformation == "clv":
            A = tf.Variable(tf.zeros((self.Dx+1, self.Dx)))
            g = tf.Variable(tf.zeros((self.Dx, )))
            Wg = tf.Variable(tf.zeros((self.Dev, self.Dx)))
            W1 = tf.Variable(tf.zeros((self.Dev, self.Dx)))
            W2 = tf.Variable(tf.zeros((self.Dx+1, 1)))
            self.f_tran = clv_transformation(params=(A, g, Wg, W1, W2))

        self.q0_tran = MLP_transformation(self.q0_layers, self.Dx,
                                          output_cov=self.output_cov,
                                          diag_cov=self.diag_cov,
                                          name="q0_tran")

        if self.use_2_q:
            self.q2_tran = MLP_transformation(self.q2_layers, self.Dx,
                                              output_cov=self.output_cov,
                                              diag_cov=self.diag_cov,
                                              name="q2_tran")
        else:
            self.q2_tran = None

        if self.PSVO:
            self.BSim_q_init_tran = MLP_transformation(self.q0_layers, self.Dx,
                                                       output_cov=self.output_cov,
                                                       diag_cov=self.diag_cov,
                                                       name="BSim_q_init_tran")

            self.q1_inv_tran = MLP_transformation(self.q1_layers, self.Dx,
                                                  output_cov=self.output_cov,
                                                  diag_cov=self.diag_cov,
                                                  name="q1_inv_tran")
            self.BSim_q2_tran = MLP_transformation(self.q2_layers, self.Dx,
                                                   output_cov=self.output_cov,
                                                   diag_cov=self.diag_cov,
                                                   name="BSim_q2_tran")

        if self.use_bootstrap:
            self.q1_tran = self.f_tran
        else:
            self.q1_tran = MLP_transformation(self.f_layers, self.Dx,
                                             output_cov=self.output_cov,
                                             diag_cov=self.diag_cov,
                                             name="f_tran")

        if self.two_step_emission:
            self.h_tran = MLP_transformation(self.h_layers, self.Dx,
                                             output_cov=self.output_cov,
                                             diag_cov=self.diag_cov,
                                             name="h_tran")


        self.g_tran = MLP_transformation(self.g_layers, self.Dy,
                                         output_cov=self.output_cov,
                                         diag_cov=self.diag_cov,
                                         name="g_tran")

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

        if self.two_step_emission:
            self.h_dist = tf_mvn(self.h_tran, name="h_dist",
                                 sigma_init=self.h_sigma_init,
                                 sigma_min=self.h_sigma_min)

        if self.emission == "mvn":
            self.g_dist = tf_mvn(self.g_tran, name="g_dist",
                                 sigma_init=self.g_sigma_init,
                                 sigma_min=self.g_sigma_min)
        else:
            self.g_dist = SUPPORTED_EMISSION[self.emission](self.g_tran, name="g_dist")

    def init_RNNs(self):
        if self.SVO or self.PSVO:
            y_smoother_f = [tf.contrib.rnn.LSTMBlockCell(Dh, name="y_smoother_f_{}".format(i))
                            for i, Dh in enumerate(self.y_smoother_Dhs)]
            y_smoother_b = [tf.contrib.rnn.LSTMBlockCell(Dh, name="y_smoother_b_{}".format(i))
                            for i, Dh in enumerate(self.y_smoother_Dhs)]
            if not self.use_stack_rnn:
                y_smoother_f = tf.nn.rnn_cell.MultiRNNCell(y_smoother_f)
                y_smoother_b = tf.nn.rnn_cell.MultiRNNCell(y_smoother_b)

            if self.X0_use_separate_RNN:
                X0_smoother_f = [tf.contrib.rnn.LSTMBlockCell(Dh, name="X0_smoother_f_{}".format(i))
                                 for i, Dh in enumerate(self.X0_smoother_Dhs)]
                X0_smoother_b = [tf.contrib.rnn.LSTMBlockCell(Dh, name="X0_smoother_b_{}".format(i))
                                 for i, Dh in enumerate(self.X0_smoother_Dhs)]
                if not self.use_stack_rnn:
                    X0_smoother_f = tf.nn.rnn_cell.MultiRNNCell(X0_smoother_f)
                    X0_smoother_b = tf.nn.rnn_cell.MultiRNNCell(X0_smoother_b)
            else:
                X0_smoother_f = X0_smoother_b = None

            self.bRNN = (y_smoother_f, y_smoother_b, X0_smoother_f, X0_smoother_b)

        else:
            self.bRNN = None

        if not (self.use_bootstrap and self.use_2_q):
            self.X0_transformer = Dense(self.Dx,
                                        activation="linear",
                                        kernel_initializer="he_uniform",
                                        name="X0_transformer")

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


