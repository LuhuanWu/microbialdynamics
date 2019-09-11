import tensorflow as tf
from tensorflow.keras.layers import Dense

from src.transformation.flow import NF
from src.transformation.MLP import MLP_transformation
from src.distribution.mvn import tf_mvn
from src.distribution.poisson import tf_poisson
from src.distribution.dirichlet import tf_dirichlet

SUPPORTED_EMISSION = dict(dirichlet=tf_dirichlet, poisson=tf_poisson, mvn=tf_mvn)


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

        self.Dx_encoded = FLAGS.Dx_encoded
        self.Dv_encoded = FLAGS.Dv_encoded

        self.batch_size = FLAGS.batch_size

        # Feed-Forward Network (FFN) architectures
         #initial proposal
        self.q0_layers = [int(x) for x in FLAGS.q0_layers.split(",")]
        # proposal
        self.q1_layers = [int(x) for x in FLAGS.q1_layers.split(",")]

        self.f_layers  = [int(x) for x in FLAGS.f_layers.split(",")]
        self.g_layers  = [int(x) for x in FLAGS.g_layers.split(",")]

        self.q0_sigma_init, self.q0_sigma_min = FLAGS.q0_sigma_init, FLAGS.q0_sigma_min
        self.q1_sigma_init, self.q1_sigma_min = FLAGS.q1_sigma_init, FLAGS.q1_sigma_min
        self.f_sigma_init,  self.f_sigma_min  = FLAGS.f_sigma_init, FLAGS.f_sigma_min
        self.g_sigma_init,  self.g_sigma_min  = FLAGS.f_sigma_init, FLAGS.g_sigma_min

        # bidirectional RNN architectures
        self.y_smoother_Dhs  = [int(x) for x in FLAGS.y_smoother_Dhs.split(",")]
        self.X0_smoother_Dhs = [int(x) for x in FLAGS.X0_smoother_Dhs.split(",")]

        self.output_cov                = FLAGS.output_cov
        self.diag_cov                  = FLAGS.diag_cov

        self.emission                  = FLAGS.emission  # type of emission

        self.X0_use_separate_RNN       = FLAGS.X0_use_separate_RNN
        self.use_stack_rnn             = FLAGS.use_stack_rnn

        self.init_placeholder()
        self.init_trans()
        self.init_dist()
        self.init_RNNs()

    def init_placeholder(self):
        self.obs = tf.placeholder(tf.float32, shape=(self.batch_size, None, self.Dy), name="obs")
        self.hidden = tf.placeholder(tf.float32, shape=(self.batch_size, None, self.Dx), name="hidden")
        self.input = tf.placeholder(tf.float32, shape=(self.batch_size, None, self.Dv), name="input")
        self.time = tf.placeholder(tf.int32, shape=(), name="time")
        self.mask = tf.placeholder(tf.bool, shape=(self.batch_size, None), name="mask")
        self.time_interval = tf.placeholder(tf.float32, shape=(self.batch_size, None), name="time_interval")
        self.extra_inputs = tf.placeholder(tf.float32, shape=(self.batch_size, None), name="extra_inputs")

    def init_trans(self):
        self.q0_tran = MLP_transformation(self.q0_layers, self.Dx,
                                          output_cov=self.output_cov,
                                          diag_cov=self.diag_cov,
                                          name="q0_tran")
        # transition: q1(*|h_t-1, smooth_yt)
        self.q1_tran = MLP_transformation(self.q1_layers, self.Dx,
                                          output_cov=self.output_cov,
                                          diag_cov=self.diag_cov,
                                          name="q1_tran")

        self.f_tran = MLP_transformation(self.f_layers, self.Dx,
                                         output_cov=self.output_cov,
                                         diag_cov=self.diag_cov,
                                         name="f_tran")

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

        self.f_dist = tf_mvn(self.f_tran,
                             sigma_init=self.f_sigma_init,
                             sigma_min=self.f_sigma_min,
                             name="f_dist")

        if self.emission == "mvn":
            self.g_dist = tf_mvn(self.g_tran, name="g_dist",
                                 sigma_init=self.g_sigma_init,
                                 sigma_min=self.g_sigma_min)
        else:
            self.g_dist = SUPPORTED_EMISSION[self.emission](self.g_tran, name="g_dist")

    def init_RNNs(self):

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

