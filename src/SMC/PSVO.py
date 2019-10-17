import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from src.SMC.SVO import SVO


class PSVO(SVO):
    def __init__(self, model, FLAGS, name="log_ZSMC"):
        SVO.__init__(self, model, FLAGS, name="log_ZSMC")

        self.n_particles_for_BSim_proposal = FLAGS.n_particles_for_BSim_proposal

        self.smooth_obs = False
        self.BSim_use_single_RNN = FLAGS.BSim_use_single_RNN

        self.q1_inv = model.q1_inv_dist
        self.BSim_q_init = model.Bsim_q_init_dist
        self.BSim_q2 = model.BSim_q2_dist

    def get_log_ZSMC(self, obs, hidden, input, time, mask, time_interval, extra_inputs):
        """
        Get log_ZSMC from obs y_1:T
        Inputs are all placeholders
            obs.shape = (batch_size, ?, Dy)
            hidden.shape = (batch_size, ?, Dx)
            input.shape = (batch_size, ?, Dv)
            time.shape = ()
            where ? = time
            mask.shape = (batch_size, time)
        Output:
            log_ZSMC: shape = ()
            log: stuff to debug
        """
        self.batch_size, _, _ = obs.get_shape().as_list()
        self.Dx = self.model.Dx
        self.time = time

        with tf.variable_scope(self.name):

            log = {}

            # get X_1:T, resampled X_1:T and log(W_1:T) from SMC
            X_prevs, _, log_Ws = self.SMC(hidden, obs, input, mask, time_interval, extra_inputs)
            bw_Xs, f_log_probs, g_log_probs, bw_log_Omega = \
                self.backward_simulation_w_proposal(X_prevs, log_Ws, obs, input, mask, time_interval, extra_inputs)

            log_ZSMC = self.compute_log_ZSMC(f_log_probs, g_log_probs, bw_log_Omega)
            Xs = bw_Xs

            # shape = (batch_size, time, n_particles, Dx)
            Xs = tf.transpose(Xs, perm=[2, 0, 1, 3], name="Xs")

            log["Xs"] = Xs

        return log_ZSMC, log

    @staticmethod
    def compute_log_ZSMC(f_log_probs, g_log_probs, bw_log_Omegas):
        """
        Input:
            bw_log_Ws.shape = (time, n_particles, batch_size)
            f_log_probs.shape = (time, n_particles, batch_size)
            g_log_probs.shape = (time, n_particles, batch_size)
        """
        time, n_particles, batch_size = f_log_probs.get_shape().as_list()

        joint = tf.reduce_sum(f_log_probs + g_log_probs, axis=0)
        proposal = tf.reduce_sum(bw_log_Omegas, axis=0)
        log_ZSMC = tf.reduce_logsumexp(joint - proposal, axis=0) - tf.log(tf.constant(n_particles, dtype=tf.float32))
        log_ZSMC = tf.reduce_mean(log_ZSMC)

        return log_ZSMC

    def backward_simulation_w_proposal(self, Xs, log_Ws, obs, input, mask, time_interval, extra_inputs):
        Dx, time, n_particles, batch_size = self.Dx, self.time, self.n_particles, self.batch_size

        assert batch_size == 1

        M = self.n_particles_for_BSim_proposal

        # store in reverse order
        bw_Xs_ta = tf.TensorArray(tf.float32, size=time, name="backward_X_ta")
        f_log_probs_ta = tf.TensorArray(tf.float32, size=time, name="joint_f_log_probs")
        g_log_probs_ta = tf.TensorArray(tf.float32, size=time, name="joint_g_log_probs")
        bw_log_Omegas_ta = tf.TensorArray(tf.float32, size=time, name="bw_log_Omegas_ta")

        if self.model.emission == "poisson" or self.model.emission == "multinomial" or self.model.emission == "mvn" :
            # transform to percentage
            obs_4_proposal = obs / tf.reduce_sum(obs, axis=-1, keepdims=True)
        else:
            obs_4_proposal = obs
        if self.log_dynamics:
            obs_4_proposal = tf.log(obs_4_proposal)
        elif self.lar_dynamics:
            obs_4_proposal = self.lar_transform(obs_4_proposal)
        preprocessed_X0, preprocessed_obs = self.BS_preprocess_obs(obs_4_proposal, time_interval)

        # t = T - 1
        # proposal q(x_T | y_{1:T})
        # bw_X_Tm1.shape = (M, n_particles, batch_size, Dx)
        # bw_q_log_prob.shape = (M, n_particles, batch_size)
        bw_X_Tm1, bw_q_log_prob = \
            self.BSim_q_init.sample_and_log_prob(preprocessed_obs[-1], sample_shape=(M, n_particles))

        bw_X_Tm1_tiled = tf.tile(tf.expand_dims(bw_X_Tm1, axis=2), (1, 1, n_particles, 1, 1))
        Input = tf.tile(tf.expand_dims(input[:, time - 1, :], axis=0), (n_particles, 1, 1))  # (n_particles, batch_size, Dev)
        f_Tm2_feed = tf.concat((Xs[time - 2], Input), axis=-1)
        f_Tm2_log_prob = self.f.log_prob(f_Tm2_feed, bw_X_Tm1_tiled, Dx=self.Dx)  # (M, n_particles, n_particles, batch_size)
        g_Tm1_log_prob = self.g.log_prob(bw_X_Tm1, obs[:, time - 1], extra_inputs=extra_inputs[:, time - 1])  # (M, n_particles, batch_size)

        log_W_Tm2 = log_Ws[time - 2] - tf.reduce_logsumexp(log_Ws[time - 2], axis=0)  # (n_particles, batch_size)
        log_W_Tm1 = tf.reduce_logsumexp(f_Tm2_log_prob + log_W_Tm2, axis=2)

        bw_log_omega_Tm1 = log_W_Tm1 + g_Tm1_log_prob - bw_q_log_prob  # (n_particles, batch_size)
        bw_log_omega_Tm1 = bw_log_omega_Tm1 - tf.reduce_logsumexp(bw_log_omega_Tm1, axis=0, keepdims=True)

        bw_X_Tm1, bw_log_omega_Tm1, g_Tm1_log_prob, bw_q_log_prob = \
            self.resample_X([bw_X_Tm1, bw_log_omega_Tm1, g_Tm1_log_prob, bw_q_log_prob],
                            bw_log_omega_Tm1,
                            sample_size=())

        bw_log_Omega_Tm1 = bw_log_omega_Tm1 + bw_q_log_prob + tf.log(float(M))

        bw_Xs_ta = bw_Xs_ta.write(time - 1, bw_X_Tm1)
        g_log_probs_ta = g_log_probs_ta.write(time - 1, g_Tm1_log_prob)
        bw_log_Omegas_ta = bw_log_Omegas_ta.write(time - 1, bw_log_Omega_Tm1)
        preprocessed_obs_ta = \
            tf.TensorArray(tf.float32, size=time, name="preprocessed_obs_ta").unstack(preprocessed_obs)

        #  from t = T - 2 to 1
        def while_cond(t, *unused_args):
            return t >= 1

        def while_body(t, bw_X_tp1, bw_Xs_ta, f_log_probs_ta, g_log_probs_ta, bw_log_Omegas_ta):
            # proposal q(x_t | x_t+1, y_{1:T})
            # bw_X_t.shape = (M, n_particles, batch_size, Dx)
            # bw_q_log_prob.shape = (M, n_particles, batch_size)
            Input = tf.tile(tf.expand_dims(input[:, t, :], axis=0),
                            (n_particles, 1, 1))  # (n_particles, batch_size, Dev)
            q1_inv_feed = tf.concat((bw_X_tp1, Input), axis=-1)
            bw_X_t, bw_q_log_prob, _ = self.sample_from_2_dist(self.q1_inv, self.BSim_q2,
                                                               q1_inv_feed, preprocessed_obs_ta.read(t),
                                                               sample_size=M)

            # f(x_t+1 | x_t) (M, n_particles, batch_size)
            Input = tf.tile(input[:, t, :][tf.newaxis, tf.newaxis, :, :],
                            (M, n_particles, 1, 1))  # (M, n_particles, batch_size, Dev)
            f_t_feed = tf.concat((bw_X_t, Input), axis=-1)
            f_t_log_prob = self.f.log_prob(f_t_feed, bw_X_tp1, Dx=self.Dx)

            # p(x_t | y_{1:t}) is proprotional to \int p(x_t-1 | y_{1:t-1}) * f(x_t | x_t-1) dx_t-1 * g(y_t | x_t)
            bw_X_t_tiled = tf.tile(tf.expand_dims(bw_X_t, axis=2), (1, 1, n_particles, 1, 1))
            Input = tf.tile(input[:, t, :][tf.newaxis, :, :], (n_particles, 1, 1))  # (n_particles, batch_size, Dev)
            f_tm1_feed = tf.concat((Xs[t - 1], Input), axis=-1)
            f_tm1_log_prob = self.f.log_prob(f_tm1_feed, bw_X_t_tiled, Dx=self.Dx)  # (M, n_particles, n_particles, batch_size)
            g_t_log_prob = self.g.log_prob(bw_X_t, obs[:, t], extra_inputs=extra_inputs[:, t])  # (M, n_particles, batch_size)

            log_W_tm1 = log_Ws[t - 1] - tf.reduce_logsumexp(log_Ws[t - 1], axis=0)
            log_W_t = tf.reduce_logsumexp(f_tm1_log_prob + log_W_tm1, axis=2)

            # p(x_t | x_{t+1:T}, y_{1:T})
            bw_log_omega_t = log_W_t + f_t_log_prob + g_t_log_prob - bw_q_log_prob
            bw_log_omega_t = bw_log_omega_t - tf.reduce_logsumexp(bw_log_omega_t, axis=0)

            bw_X_t, bw_log_omega_t, f_t_log_prob, g_t_log_prob, bw_q_log_prob = \
                self.resample_X([bw_X_t, bw_log_omega_t, f_t_log_prob, g_t_log_prob, bw_q_log_prob],
                                bw_log_omega_t,
                                sample_size=())

            bw_log_Omega_t = bw_log_omega_t + bw_q_log_prob + tf.log(float(M))

            bw_Xs_ta = bw_Xs_ta.write(t, bw_X_t)
            f_log_probs_ta = f_log_probs_ta.write(t + 1, f_t_log_prob)
            g_log_probs_ta = g_log_probs_ta.write(t, g_t_log_prob)
            bw_log_Omegas_ta = bw_log_Omegas_ta.write(t, bw_log_Omega_t)

            return t - 1, bw_X_t, bw_Xs_ta, f_log_probs_ta, g_log_probs_ta, bw_log_Omegas_ta

        # conduct the while loop
        init_state = (time - 2, bw_X_Tm1, bw_Xs_ta, f_log_probs_ta, g_log_probs_ta, bw_log_Omegas_ta)
        t, bw_X_1, bw_Xs_ta, f_log_probs_ta, g_log_probs_ta, bw_log_Omegas_ta = \
            tf.while_loop(while_cond, while_body, init_state)

        # t = 0
        # bw_X_t.shape = (M, n_particles, batch_size, Dx)
        # bw_q_log_prob.shape = (M, n_particles, batch_size)
        Input = tf.tile(tf.expand_dims(input[:, t, :], axis=0), (n_particles, 1, 1))  # (n_particles, batch_size, Dev)
        q1_inv_feed = tf.concat((bw_X_1, Input), axis=-1)
        bw_X_0, bw_q_log_prob, _ = self.sample_from_2_dist(self.q1_inv, self.BSim_q2,
                                                           q1_inv_feed, preprocessed_obs[0],
                                                           sample_size=M)
        Input = tf.tile(input[:, t, :][tf.newaxis, tf.newaxis, :, :],
                        (M, n_particles, 1, 1))  # (M, n_particles, batch_size, Dev)
        f_0_feed = tf.concat((bw_X_0, Input), axis=-1)
        f_0_log_prob = self.f.log_prob(f_0_feed, bw_X_1, Dx=self.Dx)  # (M, n_particles, batch_size)
        g_0_log_prob = self.g.log_prob(bw_X_0, obs[:, 0], extra_inputs=extra_inputs[:, 0])  # (M, n_particles, batch_size)

        # self.preprocessed_X0_f is cached in self.SMC()
        mu_0 = self.preprocessed_X0
        if not (self.model.use_bootstrap and self.model.use_2_q):
            f_init_log_prob = self.f.log_prob(mu_0, bw_X_0, Dx=self.Dx)  # (M, n_particles, batch_size)
        else:
            f_init_log_prob = self.q0.log_prob(mu_0, bw_X_0, Dx=self.Dx)  # (M, n_particles, batch_size)

        log_W_0 = f_init_log_prob

        bw_log_omega_0 = log_W_0 + f_0_log_prob + g_0_log_prob - bw_q_log_prob
        bw_log_omega_0 = bw_log_omega_0 - tf.reduce_logsumexp(bw_log_omega_0, axis=0)
        bw_X_0, bw_log_omega_0, f_0_log_prob, f_init_log_prob, g_0_log_prob, bw_q_log_prob = \
            self.resample_X([bw_X_0, bw_log_omega_0, f_0_log_prob, f_init_log_prob, g_0_log_prob, bw_q_log_prob],
                            bw_log_omega_0,
                            sample_size=())

        bw_log_Omegas_0 = bw_log_omega_0 + bw_q_log_prob + tf.log(float(M))

        bw_Xs_ta = bw_Xs_ta.write(0, bw_X_0)
        f_log_probs_ta = f_log_probs_ta.write(1, f_0_log_prob)
        f_log_probs_ta = f_log_probs_ta.write(0, f_init_log_prob)
        g_log_probs_ta = g_log_probs_ta.write(0, g_0_log_prob)
        bw_log_Omegas_ta = bw_log_Omegas_ta.write(0, bw_log_Omegas_0)

        # transfer tensor arrays to tensors
        bw_Xs = bw_Xs_ta.stack()
        bw_log_Omegas = bw_log_Omegas_ta.stack()
        f_log_probs = f_log_probs_ta.stack()
        g_log_probs = g_log_probs_ta.stack()

        bw_Xs.set_shape((None, n_particles, batch_size, Dx))
        bw_log_Omegas.set_shape((None, n_particles, batch_size))
        f_log_probs.set_shape((None, n_particles, batch_size))
        g_log_probs.set_shape((None, n_particles, batch_size))

        return bw_Xs, f_log_probs, g_log_probs, bw_log_Omegas

    def BS_preprocess_obs(self, obs, time_interval):
        # if self.smooth_obs, smooth obs with bidirectional RNN
        with tf.variable_scope("smooth_obs"):
            if self.BSim_use_single_RNN:
                cells = self.y_smoother_f
                if isinstance(self.y_smoother_f, list):
                    cells = tf.nn.rnn_cell.MultiRNNCell(self.y_smoother_f)
                preprocessed_obs, preprocessed_X0 = tf.nn.static_rnn(cells, tf.unstack(obs, axis=1), dtype=tf.float32)
            else:
                preprocessed_X0, preprocessed_obs = self.preprocess_obs_w_bRNN(obs, time_interval)

        return preprocessed_X0, preprocessed_obs
