import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.contrib.layers import fully_connected, xavier_initializer


class VartiationalRNN():

    def __init__(self, model, FLAGS, variable_scope="VRNN"):
        self.model = model

        self.Dh = FLAGS.Dh
        self.Dx = model.Dx
        self.Dy = model.Dy
        self.Dv = model.Dv

        # x feature extraction
        self.Dx_encoded = model.Dx_encoded
        # v feature extraction
        self.Dv_encoded = model.Dv_encoded

        self.n_particles = FLAGS.n_particles
        self.batch_size = FLAGS.batch_size
        self.variable_scope = variable_scope

        self.lstm = tf.nn.rnn_cell.LSTMCell(self.Dh, state_is_tuple=True)

    def reset(self):
        # lstm return to zero_state
        with tf.variable_scope(self.variable_scope + '/reset'):
            self.state = self.lstm.zero_state((self.n_particles * self.batch_size), dtype=tf.float32)
            c,  self.h = self.state

            # self.h_stack.shape = (n_particles, batch_size, self.Dh)
            self.h_stack = tf.reshape(self.h, (self.n_particles, self.batch_size, self.Dh), name='h_stack')
            c = tf.reshape(c, (self.n_particles, self.batch_size, self.Dh), name='c_stack')

        return c, self.h_stack

    def get_x_ft(self, x):
        # x feature extraction
        with tf.variable_scope(self.variable_scope + '/get_x_ft'):
            x_1 = fully_connected(x, self.Dx_encoded, reuse = tf.AUTO_REUSE, scope ='x_to_x_1')
            x_ft = tf.identity(x_1, name='x_ft')
        return x_ft

    def get_v_ft(self, v):
        # v feature extraction
        with tf.variable_scope(self.variable_scope + '/get_v_ft'):
            v_1 = fully_connected(v, self.Dv_encoded, reuse=tf.AUTO_REUSE, scope="v_to_v_encoded")
            v_ft = tf.identity(v_1, name="v_ft")
        return v_ft

    def get_q0_sample_and_log_prob(self, smooth_y0):
        # calculate q, q(x_0 | h_0, smooth_y0)
        # smooth_y0.shape = (batch_size, D_smooth_y0)
        D_smooth_y0 = smooth_y0.shape[-1]
        smooth_y_0_tiled = tf.tile(smooth_y0[None,], (self.n_particles, 1, 1),
                               name='smooth_yt_tiled')  # (n_paticles, batch_size, D_smooth_y0)
        h_y_concat = tf.concat((self.h_stack, smooth_y_0_tiled), axis=-1,
                               name='h_y_concat')  # (n_paticles, batch_size, Dh + D_smooth_y0)

        assert h_y_concat.shape == (self.n_particles, self.batch_size, self.Dh + D_smooth_y0)

        X, q_0_log_prob = self.model.q0_dist.sample_and_log_prob(h_y_concat,
                                                                 name="q_t_sample_and_log_prob")
        return X, q_0_log_prob

    def get_q_sample_and_log_prob(self, smooth_yt):
        # calculate q, q(x_t | h_t, smooth_yt)
        smooth_y_t_tiled = tf.tile(smooth_yt[None,], (self.n_particles, 1, 1),
                               name='smooth_yt_tiled')  # (n_paticles, batch_size, D_smooth_yt)
        h_y_concat = tf.concat((self.h_stack, smooth_y_t_tiled), axis=2,
                               name='h_y_concat')  # (n_paticles, batch_size, Dh + D_smooth_yt)

        X, q_t_log_prob = self.model.q1_dist.sample_and_log_prob(h_y_concat,
                                               name="q1_t_sample_and_log_prob")
        return X, q_t_log_prob

    def get_f_log_prob(self, x):
        """
        calculate f(x_t|h_t)
        x.shape = (n_particles, batch_size, Dx)
        """

        f_t_log_prob = self.model.f_dist.log_prob(self.h_stack, x, name="f_t_log_prob")
        return f_t_log_prob

    def get_f_mean(self, lstm_states):
        """

        :param lstm_states: (time, batch_size, Dh)
        :return:
        """
        return self.model.f_dist.mean(lstm_states)

    def get_g_log_prob(self, x_t, y_t, **kwargs):
        """
        calculate g(y_t|h_t, x_t)
        x_t.shape = (n_particles, batch_size, Dx)
        y_t.shape = (batch_size, Dy)
        """

        # prepare conditioning input
        x_t_ft = self.get_x_ft(x_t)
        assert x_t_ft.shape == (self.n_particles, self.batch_size, self.Dx_encoded)

        h_x_concat = tf.concat((self.h_stack, x_t_ft), axis=-1, name="h_x_t_concat")

        g_t_log_prob = self.model.g_dist.log_prob(h_x_concat, y_t, name="g_t_log_prob", **kwargs)
        return g_t_log_prob

    def get_g_mean(self, lstm_states, x, **kwargs):
        """
        g(y_t | h_t, x_t)
        :param lstm_states: (batch_size, time, Dh)
        :param x: (batch_size, time, Dx)
        :param kwargs:
        :return:
        """
        x_ft = self.get_x_ft(x)

        h_x_concat = tf.concat((lstm_states, x_ft), axis=-1, name="h_x_concat") # (batch_size, time, Dh + Dx_encoded)

        return self.model.g_dist.mean(h_x_concat, **kwargs)

    def update_lstm(self, x_t, v_t):
        """
        update h_t to h_{t+1}
        x_t.shape = (n_particles, batch_size, Dx)
        v_t.shape = (batch_size, Dv)
        """
        x_t_ft = self.get_x_ft(x_t)  # (n_particles, batch_size, Dx_encoded)
        v_t_ft = self.get_v_ft(v_t)  # (batch_size, Dv_encoded)

        with tf.variable_scope(self.variable_scope + '/update_lstm'):
            v_t_ft_tiled = tf.tile(v_t_ft[None, ], (self.n_particles, 1, 1), name='v_t_ft_tilwd')

            xv_t_ft = tf.concat((x_t_ft, v_t_ft_tiled), axis=-1, name = 'xv_t_ft')
            # xy_t_ft.shape 		= (n_particles, batch_size, Dx_encoded + Dv_encoded)
            lstm_input = tf.reshape(xv_t_ft, (self.n_particles * self.batch_size, self.Dx_encoded + self.Dv_encoded))

            _, self.state = self.lstm(lstm_input, self.state)
            c, self.h = self.state

            # (n_particles, batch_size, self.Dh)
            self.h_stack = tf.reshape(self.h, (self.n_particles, self.batch_size, self.Dh), name='h_stack')
            c = tf.reshape(c, (self.n_particles, self.batch_size, self.Dh), name="c_stack")

        return c, self.h_stack

    def compute_lstm(self, x_t, v_t, h_t, c_t):
        """
        compute h_{t+1} using x_t, v_t, h_t
        x_t.shape = (batch_size, time, Dx)
        v_t.shape = (batch_size, time, Dv)
        h_t.shape = (batch_size, time, Dh)
        c_t.shae = (batch_size, time, Dh)
        """
        assert len(x_t.shape) == 3
        assert len(h_t.shape) == 3

        x_t_ft = self.get_x_ft(x_t)  # (batch_size, time, Dx_encoded)
        v_t_ft = self.get_v_ft(v_t)  # (batch_size, time, Dv_encoded)

        with tf.variable_scope(self.variable_scope + '/compute_lstm'):

            xv_t_ft = tf.concat((x_t_ft, v_t_ft), axis=-1, name='xv_t_ft')  # (batch_size, time, Dx_encoded + Dv_encoded)
            lstm_input = tf.reshape(xv_t_ft, (-1, self.Dx_encoded + self.Dv_encoded))

            h_t = tf.reshape(h_t, (-1, self.Dh))
            c_t = tf.reshape(c_t, (-1, self.Dh))

            _, state = self.lstm(lstm_input, (c_t, h_t))
            c_tplus1, h_tplus1 = state

            # (self.batch_size, time ,self.Dh)
            h_tplus1 = tf.reshape(h_tplus1, (self.batch_size, -1, self.Dh))
            c_tplus1 = tf.reshape(c_tplus1, (self.batch_size, -1, self.Dh))

        return c_tplus1, h_tplus1


