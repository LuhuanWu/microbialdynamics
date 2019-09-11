import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd


class VRNNSVO:
    def __init__(self, rnn_cell, FLAGS, name="log_ZSMC"):

        self.rnn_cell = rnn_cell
        self.FLAGS = FLAGS

        self.Dh = self.rnn_cell.Dh
        self.Dx = self.rnn_cell.Dx
        self.Dy = self.rnn_cell.Dy
        self.Dv = self.rnn_cell.Dv

        self.n_particles = FLAGS.n_particles

        # bidirectional RNN as full sequence observations encoder
        self.X0_use_separate_RNN = FLAGS.X0_use_separate_RNN
        self.use_stack_rnn = FLAGS.use_stack_rnn

        self.log_dynamics = FLAGS.log_dynamics
        self.smooth_obs = True
        self.resample_particles = True

        self.name = name

    def get_log_ZSMC(self, obs, hidden, input, time, mask, time_interval, extra_inputs):
        """
        Get log_ZSMC from obs y_1:T
        Inputs are all placeholders:
            obs.shape = (batch_size, time, Dy)
            hidden.shape = (batch_size, time, Dz)
            input.shape = (batch_size, time, Dx)
            time.shape = ()
            mask.shape = (batch_size, time) | batch_size=1
            time_interval:
            extra_inputs.shape = (batch_size, time) -- count data
        Output:
            log_ZSMC: shape = scalar
            log: stuff to debug
        """
        self.batch_size, _, _ = obs.get_shape().as_list()
        self.time = time

        with tf.variable_scope(self.name):

            log = {}

            # get X_1:T, resampled X_1:T and log(W_1:T) from SMC
            X_prevs, X_ancestors, log_Ws, lstm_states_h, lstm_states_c = self.SMC(hidden, obs, input, mask, time_interval, extra_inputs)
            log_ZSMC = self.compute_log_ZSMC(log_Ws)

            # shape = (batch_size, time, n_particles, Dx)
            X_ancestors = tf.transpose(X_ancestors, perm=[2, 0, 1, 3], name="Xs")

            # (batch_size, time, n_particles, Dh)
            lstm_states_h = tf.transpose(lstm_states_h, perm=[2, 0, 1, 3], name="lstm_states_h")
            lstm_states_c = tf.transpose(lstm_states_c, perm=[2, 0, 1, 3], name="lstm_states_c")

            log["Xs"] = X_ancestors
            log["lstm_states_h"] = lstm_states_h
            log["lstm_states_c"] = lstm_states_c

        return log_ZSMC, log

    def SMC(self, hidden, obs, input, mask, time_interval, extra_inputs, q_cov=1.0):
        # hidden only useful when using true X

        Dh, Dx, n_particles, batch_size, time = self.Dh, self.Dx, self.n_particles, self.batch_size, self.time

        # preprocessing obs
        if self.log_dynamics:
            log_obs = tf.log(obs)
            preprocessed_X0, preprocessed_obs = self.preprocess_obs(log_obs, time_interval)
        else:
            preprocessed_X0, preprocessed_obs = self.preprocess_obs(obs, time_interval)
        self.preprocessed_X0  = preprocessed_X0
        self.preprocessed_obs = preprocessed_obs

        # -------------------------------------- t = 0 -------------------------------------- #
        c_0, h_0 = self.rnn_cell.reset()  # (n_particles, batch_size, Dh)

        # proposal
        X, q_0_log_prob = self.rnn_cell.get_q0_sample_and_log_prob(preprocessed_X0)
        assert X.shape == (n_particles, batch_size, Dx), X.shape
        assert q_0_log_prob.shape == (n_particles, batch_size), q_0_log_prob.shape

        f_0_log_prob = self.rnn_cell.get_f_log_prob(X)
        assert f_0_log_prob.shape == (n_particles, batch_size), f_0_log_prob.shape

        # emission log probability and log weights
        if self.log_dynamics:
            _g_0_log_prob = self.rnn_cell.get_g_log_prob(tf.exp(X), obs[:, 0], extra_inputs=extra_inputs[:, 0])
        else:
            _g_0_log_prob = self.rnn_cell.get_g_log_prob(X, obs[:, 0], extra_inputs=extra_inputs[:, 0])
        _g_0_log_prob_0 = tf.zeros_like(_g_0_log_prob)  # dummy values for missing observations

        g_0_log_prob = tf.where(mask[0][0], _g_0_log_prob, _g_0_log_prob_0, name="g_{}_log_prob".format(0))

        log_alpha_0 = tf.add(f_0_log_prob, g_0_log_prob - q_0_log_prob, name="log_alpha_{}".format(0))
        log_weight_0 = tf.add(log_alpha_0, - tf.log(tf.constant(n_particles, dtype=tf.float32)),
                              name="log_weight_{}".format(0))  # (n_particles, batch_size)

        log_normalized_weight_0 = tf.add(log_weight_0, - tf.reduce_logsumexp(log_weight_0, axis=0),
                                         name="log_noramlized_weight_{}".format(0))


        # -------------------------------------- t = 1, ..., T - 1 -------------------------------------- #
        # prepare tensor arrays
        # tensor arrays to read
        preprocessed_obs_ta = \
            tf.TensorArray(tf.float32, size=time, name="preprocessed_obs_ta").unstack(preprocessed_obs)

        # tensor arrays to write
        # particles, resampled particles (mean), log weights of particles
        log_names = ["X_prevs", "X_ancestors", "log_weights", "lstm_states_h", "lstm_states_c"]
        log = [tf.TensorArray(tf.float32, size=time, clear_after_read=False, name="{}_ta".format(name))
               for name in log_names]

        # write results for t = 0 into tensor arrays
        log[2] = log[2].write(0, log_weight_0)

        def while_cond(t, *unused_args):
            return t < time

        def while_body(t, X_prev, log_normalized_weight_tminus1, h_prev, c_prev, log):
            # resampling
            X_ancestor, h_ancestores, c_ancestors = self.resample_X(X_prev, log_normalized_weight_tminus1, sample_size=n_particles,
                                         resample_particles=self.resample_particles, hs=h_prev, cs=c_prev)

            c_t, h_t = self.rnn_cell.update_lstm(X_ancestor, input[:, t-1, :])

            # proposal
            X, q_t_log_prob = self.rnn_cell.get_q_sample_and_log_prob(preprocessed_obs_ta.read(t))
            assert X.shape == (n_particles, batch_size, Dx)
            assert q_t_log_prob.shape == (n_particles, batch_size)

            f_t_log_prob = self.rnn_cell.get_f_log_prob(X)

            # emission log probability and log weights
            if self.log_dynamics:
                _g_t_log_prob = self.rnn_cell.get_g_log_prob(tf.exp(X), obs[:, t], extra_inputs=extra_inputs[:, t])
            else:
                _g_t_log_prob = self.rnn_cell.get_g_log_prob(X, obs[:, t], extra_inputs=extra_inputs[:, t])
            _g_t_log_prob_0 = tf.zeros_like(_g_t_log_prob)
            g_t_log_prob = tf.where(mask[0][t], _g_t_log_prob, _g_t_log_prob_0, name="g_t_log_prob")

            log_alpha_t = tf.add(f_t_log_prob, g_t_log_prob - q_t_log_prob, name="log_alpha_t")

            log_weight_t = tf.add(log_alpha_t, log_normalized_weight_tminus1, name="log_weight_t")

            if self.resample_particles:
                log_normalized_weight_t = tf.negative(tf.log(tf.constant(n_particles, dtype=tf.float32,
                                                                         shape=(n_particles, batch_size),
                                                                         )), name="log_normalized_weight_t")
            else:
                log_normalized_weight_t = tf.add(log_weight_t, - tf.reduce_logsumexp(log_weight_t, axis=0),
                                                 name="log_normalized_weight_t")

            # write results in this loop to tensor arrays
            idxs = [t - 1, t - 1, t, t - 1, t - 1]
            log_contents = [X_prev, X_ancestor, log_weight_t, h_ancestores, c_ancestors]
            log = [ta.write(idx, log_content) for ta, idx, log_content in zip(log, idxs, log_contents)]

            return t + 1, X, log_normalized_weight_t, h_t, c_t, log

        # conduct the while loop
        init_state = (1, X, log_normalized_weight_0, h_0, c_0, log)
        t_final, X_T, log_normalized_weight_T, h_T, c_T, log = tf.while_loop(while_cond, while_body, init_state)

        # write final results at t = T - 1 to tensor arrays
        X_T_resampled, h_T_resampled, c_T_resampled = self.resample_X(X_T, log_normalized_weight_T,
                                                                      sample_size=n_particles,
                                                                      resample_particles=self.resample_particles,
                                                                      hs=h_T, cs=c_T)
        log[0] = log[0].write(t_final - 1, X_T)
        log[1] = log[1].write(t_final - 1, X_T_resampled)
        log[3] = log[3].write(t_final - 1, h_T_resampled)
        log[4] = log[4].write(t_final - 1, c_T_resampled)

        # convert tensor arrays to tensors
        log_shapes = [(None, n_particles, batch_size, Dx)] * 2 + [(None, n_particles, batch_size)] + \
                     [(None, n_particles, batch_size, Dh)] * 2

        log = [ta.stack(name=name) for ta, name in zip(log, log_names)]

        for tensor, shape in zip(log, log_shapes):
            tensor.set_shape(shape)

        return log

    def resample_X(self, X, log_W, sample_size=(), resample_particles=True, hs=None, cs=None):
        """
        Resample X using categorical with logits = log_W
        Input:
            X: can be a list, each element e.shape = (K, batch_size_0, ..., batch_size_last, e_dim_0, ..., e_dim_last)
            log_W.shape = (K, batch_size_0, ..., batch_size_last)
            sample_size: () or int
        """
        if resample_particles:
            if log_W.shape.as_list()[0] != 1:
                resample_idx = self.get_resample_idx(log_W, sample_size)
                X_resampled = self.resample_according_to_idx(X, resample_idx)
                if hs is not None:
                    hs_resampled = self.resample_according_to_idx(hs, resample_idx)
                if cs is not None:
                    cs_resampled = self.resample_according_to_idx(cs, resample_idx)
            else:
                assert sample_size == 1
                return X, hs, cs

        return X, hs, cs

    def resample_according_to_idx(self, X, resample_idx):
        if isinstance(X, list):
            X_resampled = [tf.gather_nd(item, resample_idx) for item in X]
        else:
            X_resampled = tf.gather_nd(X, resample_idx)
        return X_resampled

    def get_resample_idx(self, log_W, sample_size=()):
        """
        Get resample index a_t^k ~ Categorical(w_t^1, ..., w_t^K) with logits = log_W last axis
        Input:
            log_W.shape = (K, batch_size_0, ..., batch_size_last)
        """
        nb_classes  = log_W.shape.as_list()[0]
        batch_shape = log_W.shape.as_list()[1:]
        perm = list(range(1, len(batch_shape) + 1)) + [0]

        log_W = tf.transpose(log_W, perm=perm)
        categorical = tfd.Categorical(logits=log_W, validate_args=True, name="Categorical")

        # sample multiple times to remove idx out of range
        if sample_size == ():
            idx_shape = batch_shape
        else:
            assert isinstance(sample_size, int), "sample_size should be int, {} is given".format(sample_size)
            idx_shape = [sample_size] + batch_shape

        idx = tf.ones(idx_shape, dtype=tf.int32) * nb_classes
        for _ in range(1):
            fixup_idx = categorical.sample(sample_size)
            idx = tf.where(idx >= nb_classes, fixup_idx, idx)

        # if still got idx out of range, replace them with idx from uniform distribution
        final_fixup = tf.random.uniform(idx_shape, maxval=nb_classes, dtype=tf.int32)
        idx = tf.where(idx >= nb_classes, final_fixup, idx)

        batch_idx = np.meshgrid(*[range(i) for i in idx_shape], indexing='ij')
        if sample_size != ():
            batch_idx = batch_idx[1:]
        resample_idx = tf.stack([idx] + batch_idx, axis=-1)

        return resample_idx

    @staticmethod
    def compute_log_ZSMC(log_Ws):
        """
        :param log_Ws: shape (time, n_particles, batch_size)
        :return: loss, shape ()
        """
        log_ZSMC = tf.reduce_logsumexp(log_Ws, axis=1)  # (time, batch_size)
        log_ZSMC = tf.reduce_sum(tf.reduce_mean(log_ZSMC, axis=1), name="log_ZSMC")

        return log_ZSMC

    def preprocess_obs(self, obs, time_interval):
        """

        :param obs: (batch_size, time, Dy)
        :return: preprocessed_obs, a list of length time, each item is of shape (batch_size, smoother_Dhs*2)
        """
        # if self.smooth_obs, smooth obs with bidirectional RNN
        with tf.variable_scope("smooth_obs"):
            if not self.smooth_obs:
                preprocessed_obs = tf.transpose(obs, perm=[1, 0, 2])
                preprocessed_X0 = preprocessed_obs[0]
            else:
                preprocessed_X0, preprocessed_obs = self.preprocess_obs_w_bRNN(obs, time_interval)

            """
            if not (self.model.use_bootstrap and self.model.use_2_q):
                preprocessed_X0 = self.model.X0_transformer(preprocessed_X0)
            """
        return preprocessed_X0, preprocessed_obs

    def preprocess_obs_w_bRNN(self, obs, time_interval):
        self.y_smoother_f, self.y_smoother_b, self.X0_smoother_f, self.X0_smoother_b = self.rnn_cell.model.bRNN
        rnn_input = tf.concat([obs, time_interval[:, :, tf.newaxis]], axis=-1)

        if self.use_stack_rnn:
            outputs, state_fw, state_bw = \
                tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.y_smoother_f,
                                                               self.y_smoother_b,
                                                               rnn_input,
                                                               dtype=tf.float32)
        else:
            outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(self.y_smoother_f,
                                                                            self.y_smoother_b,
                                                                            rnn_input,
                                                                            dtype=tf.float32)
        smoothed_obs = tf.concat(outputs, axis=-1)
        preprocessed_obs = tf.transpose(smoothed_obs, perm=[1, 0, 2])

        if self.X0_use_separate_RNN:
            if self.use_stack_rnn:
                outputs, state_fw, state_bw = \
                    tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.X0_smoother_f,
                                                                   self.X0_smoother_b,
                                                                   rnn_input,
                                                                   dtype=tf.float32)
            else:
                outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(self.X0_smoother_f,
                                                                                self.X0_smoother_b,
                                                                                rnn_input,
                                                                                dtype=tf.float32)
        if self.use_stack_rnn:
            outputs_fw = outputs_bw = outputs
        else:
            outputs_fw, outputs_bw = outputs
        preprocessed_X0 = tf.concat([outputs_fw[:, -1, :], outputs_bw[:, 0, :]], axis=-1)

        return preprocessed_X0, preprocessed_obs

    def n_step_MSE(self, n_steps, lstm_states_h, lstm_states_c, hidden, obs, input, mask, extra_inputs):
        """
        Compute MSE_k for k = 0, ..., n_steps. This is an intermediate step to calculate k-step R^2
        :param n_steps: integer
        :param lstm_states_h: (batch_size, time, n_particles, Dh)
        :param lstm_states_c: (batch_size, time, n_particles, Dh)
        :param hidden: (batch_size, time, n_particles, Dx)
        :param obs: (batch_size, time, Dy)
        :param input: (batch_size, time, Dv)
        :param mask: (batch_size, time)
        :param extra_inputs: (batch_size, time)
        :return:
        """

        batch_size, _, _, _ = hidden.shape.as_list()

        assert batch_size == 1

        _, _, Dy = obs.shape.as_list()  # (batch_size, time, Dy)
        # assert n_steps < time, "n_steps = {} >= time".format(n_steps)

        with tf.variable_scope(self.name):

            # average over paths
            lstm_states_h = tf.reduce_mean(lstm_states_h, axis=2)  # (batch_size, time, Dh)
            lstm_states_c = tf.reduce_mean(lstm_states_c, axis=2)  # (batch_size, time, Dh)
            hidden = tf.reduce_mean(hidden, axis=2)  # (batch_size, time, Dx)
            x_BxTmkxDz = hidden
            h_BxTmkxDh = lstm_states_h
            c_BxTmkxDh = lstm_states_c

            # get y_hat
            y_hat_N_BxTxDy = []

            for k in range(n_steps):
                if self.log_dynamics:
                    y_hat_BxTmkxDy = self.rnn_cell.get_g_mean(h_BxTmkxDh, tf.exp(x_BxTmkxDz),
                                                              extra_inputs=extra_inputs[:, k:])
                else:
                    y_hat_BxTmkxDy = self.rnn_cell.get_g_mean(h_BxTmkxDh, x_BxTmkxDz,
                                                              extra_inputs=extra_inputs[:, k:])

                y_hat_N_BxTxDy.append(y_hat_BxTmkxDy)  # each item is (batch_size, time - k, Dy)

                x_BxTmkxDz = x_BxTmkxDz[:, :-1]  # (batch_size, time - k - 1, Dx)
                h_BxTmkxDh = h_BxTmkxDh[:, :-1]  # (batch_size, time - k - 1, Dh)
                c_BxTmkxDh = c_BxTmkxDh[:, :-1]  # (batch_size, time - k - 1, Dh)

                # (batch_size, time - k - 1, Dh)
                c_BxTmkxDh, h_BxTmkxDh = self.rnn_cell.compute_lstm(x_BxTmkxDz, input[:, k:-1], h_BxTmkxDh, c_BxTmkxDh)
                x_BxTmkxDz = self.rnn_cell.get_f_mean(h_BxTmkxDh) # (batch_size, time - k - 1, Dx)

            if self.log_dynamics:
                y_hat_BxTmNxDy = self.rnn_cell.get_g_mean(h_BxTmkxDh, tf.exp(x_BxTmkxDz), extra_inputs=extra_inputs[:, n_steps:])
            else:
                y_hat_BxTmNxDy = self.rnn_cell.get_g_mean(h_BxTmkxDh, x_BxTmkxDz, extra_inputs=extra_inputs[:, n_steps:])
            y_hat_N_BxTxDy.append(y_hat_BxTmNxDy) # each item is # (batch_size, T - N, Dy)

            # get y_true
            y_N_BxTxDy = []
            for k in range(n_steps + 1):
                y_BxTmkxDy = obs[:, k:, :]
                y_N_BxTxDy.append(y_BxTmkxDy)

            # compare y_hat and y_true to get MSE_k, y_mean, y_var
            # FOR THE BATCH and FOR k = 0, ..., n_steps

            MSE_ks = []     # [MSE_0, MSE_1, ..., MSE_N]
            y_means = []    # [y_mean_0 (shape = Dy), ..., y_mean_N], used to calculate y_var across all batches
            y_vars = []     # [y_var_0 (shape = Dy), ..., y_var_N], used to calculate y_var across all batches
            for k, (y_hat_BxTmkxDy, y_BxTmkxDy) in enumerate(zip(y_hat_N_BxTxDy, y_N_BxTxDy)):

                if self.FLAGS.emission == "poisson" or "multinomial":
                    # convert count into percentage
                    y_hat_BxTmkxDy = y_hat_BxTmkxDy / tf.reduce_sum(y_hat_BxTmkxDy, axis=-1, keepdims=True)
                    y_BxTmkxDy = y_BxTmkxDy / tf.reduce_sum(y_BxTmkxDy, axis=-1, keepdims=True)

                # convert percentage into log percentage
                y_hat_BxTmkxDy = (y_hat_BxTmkxDy + 1e-9) / (1 + Dy * 1e-9)
                y_hat_BxTmkxDy = tf.log(y_hat_BxTmkxDy)
                y_BxTmkxDy = (y_BxTmkxDy + 1e-9) / (1 + Dy * 1e-9)
                y_BxTmkxDy = tf.log(y_BxTmkxDy)

                difference = y_hat_BxTmkxDy - y_BxTmkxDy   # (batch_size, time-k, Dy)
                masked_difference = tf.boolean_mask(difference, mask[:, k:])  # (time-k, Dy)
                masked_difference = masked_difference[None,]  # (batch_size, time-k, Dy)

                MSE_k = tf.reduce_sum(masked_difference**2, name="MSE_{}".format(k))
                MSE_ks.append(MSE_k)

                masked_y = tf.boolean_mask(y_BxTmkxDy, mask[:, k:])   # (mask_time, Dy)
                masked_y = masked_y[None,]  # (batch_size, mask_time, Dy)
                y_mean = tf.reduce_mean(masked_y, axis=[0, 1], name="y_mean_{}".format(k))  # (Dy,)
                y_means.append(y_mean)
                y_var = tf.reduce_sum((masked_y - y_mean)**2, axis=[0, 1], name="y_var_{}".format(k))   # (Dy,)
                y_vars.append(y_var)

            MSE_ks = tf.stack(MSE_ks, name="MSE_ks")     # (n_steps + 1)
            y_means = tf.stack(y_means, name="y_means")  # (n_steps + 1, Dy)
            y_vars = tf.stack(y_vars, name="y_vars")     # (n_steps + 1, Dy)

            return MSE_ks, y_means, y_vars, y_hat_N_BxTxDy

    """
    def get_nextX(self, X):
        # TODO: fix this
        # only used for drawing 2D quiver plot
        with tf.variable_scope(self.name):
            return self.f.mean(X)
    """
