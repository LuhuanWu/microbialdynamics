import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd


class SVO:
    def __init__(self, model, FLAGS, name="log_ZSMC"):

        self.model = model

        # SSM distributions
        self.q0 = model.q0_dist
        self.q1 = model.q1_dist
        self.q2 = model.q2_dist
        self.f  = model.f_dist
        self.g  = model.g_dist

        self.n_particles = FLAGS.n_particles
        self.q_uses_true_X = FLAGS.q_uses_true_X

        self.emission = FLAGS.emission
        self.two_step_emission = FLAGS.two_step_emission
        if self.two_step_emission:
            self.h = model.h_dist

        # bidirectional RNN as full sequence observations encoder
        self.X0_use_separate_RNN = FLAGS.X0_use_separate_RNN
        self.use_stack_rnn = FLAGS.use_stack_rnn

        self.model = model

        self.log_dynamics = FLAGS.log_dynamics
        self.lar_dynamics = FLAGS.lar_dynamics

        self.smooth_obs = True
        self.resample_particles = True

        self.f_power = FLAGS.f_power # power for q_1 and f

        self.name = name

        self.use_mask = FLAGS.use_mask

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
        self.Dx = self.model.Dx
        self.time = time

        with tf.variable_scope(self.name):

            log = {}

            # get X_1:T, resampled X_1:T and log(W_1:T) from SMC
            X_prevs, X_ancestors, log_Ws = self.SMC(hidden, obs, input, mask, time_interval, extra_inputs)
            log_ZSMC = self.compute_log_ZSMC(log_Ws)
            Xs = X_ancestors

            # shape = (batch_size, time, n_particles, Dx)
            Xs = tf.transpose(Xs, perm=[2, 0, 1, 3], name="Xs")

            log["Xs"] = Xs

        return log_ZSMC, log

    def SMC(self, hidden, obs, input, mask, time_interval, extra_inputs, q_cov=1.0):
        Dx, n_particles, batch_size, time = self.Dx, self.n_particles, self.batch_size, self.time

        # preprocessing obs
        if self.model.emission == "poisson" or self.model.emission == "multinomial" or self.model.emission == "mvn" :
            # transform to percentage
            obs_4_proposal = obs / tf.reduce_sum(obs, axis=-1, keepdims=True)
        else:
            obs_4_proposal = obs
        if self.log_dynamics:
            obs_4_proposal = tf.log(obs_4_proposal)
        elif self.lar_dynamics:
            obs_4_proposal = self.lar_transform(obs_4_proposal)
        self.preprocessed_X0, self.preprocessed_obs = self.preprocess_obs(obs_4_proposal, time_interval)
        q0, q1, f = self.q0, self.q1, self.f

        # -------------------------------------- t = 0 -------------------------------------- #
        q_f_0_feed = self.preprocessed_X0

        # proposal
        if self.q_uses_true_X:
            X_0, q_0_log_prob = self.sample_from_true_X(hidden[:, 0, :],
                                                      q_cov,
                                                      sample_shape=n_particles,
                                                      name="q_0_sample_and_log_prob")
            f_0_log_prob = f.log_prob(q_f_0_feed, X_0, name="f_0_log_prob")
        else:
            if self.model.use_2_q:
                X_0, q_0_log_prob, f_0_log_prob = self.sample_from_2_dist(q0,
                                                                        self.q2,
                                                                        q_f_0_feed,
                                                                        self.preprocessed_obs[0],
                                                                        sample_size=n_particles)
            else:
                X_0, q_0_log_prob = q0.sample_and_log_prob(q_f_0_feed,
                                                         sample_shape=n_particles,
                                                         name="q_0_sample_and_log_prob")

                # only when use_bootstrap and use_2_q, f_t_log_prob has been calculated
                assert not self.model.use_bootstrap
                f_0_log_prob = f.log_prob(q_f_0_feed, X, name="f_0_log_prob", Dx=self.Dx)

        # emission log probability and log weights
        if self.log_dynamics or self.lar_dynamics:
            g_input = tf.exp(X_0)
        else:
            g_input = X_0
        if self.two_step_emission:
            g_input, _h_0_log_prob = self.h.sample_and_log_prob(g_input)
            if self.use_mask:
                _h_0_log_prob_0 = tf.zeros_like(_h_0_log_prob)  # dummy values for missing observations
                h_0_log_prob = tf.where(mask[0][0], _h_0_log_prob, _h_0_log_prob_0, name="h_{}_log_prob".format(0))
            else:
                h_0_log_prob = _h_0_log_prob
        _g_0_log_prob = self.g.log_prob(g_input, obs[:, 0], extra_inputs=extra_inputs[:, 0])
        _g_0_log_prob_0 = tf.zeros_like(_g_0_log_prob)  # dummy values for missing observations
        if self.use_mask:
            g_0_log_prob = tf.where(mask[0][0], _g_0_log_prob, _g_0_log_prob_0, name="g_{}_log_prob".format(0))
        else:
            g_0_log_prob = _g_0_log_prob

        log_alpha_0 = tf.add(f_0_log_prob, g_0_log_prob - q_0_log_prob, name="log_alpha_{}".format(0))
        if self.two_step_emission:
            log_alpha_0 += h_0_log_prob - tf.stop_gradient(h_0_log_prob)

        log_W_0 = tf.add(log_alpha_0, - tf.log(tf.constant(n_particles, dtype=tf.float32)),
                              name="log_W_0")  # (n_particles, batch_size)
        X_ancestor_0 = self.resample_X(X_0, log_W_0, sample_size=n_particles,
                                       resample_particles=self.resample_particles)
        if self.resample_particles:
            log_normalized_W_0 = tf.negative(tf.log(tf.constant(n_particles, dtype=tf.float32,
                                                          shape=(n_particles, batch_size))),
                                              name="log_normalized_W_0")

        else:
            log_normalized_W_0 = tf.add(log_W_0, - tf.reduce_logsumexp(log_W_0, axis=0),
                                         name="log_normalized_W_{}".format(0))


        # -------------------------------------- t = 1, ..., T - 1 -------------------------------------- #
        Xs_ta = tf.TensorArray(tf.float32, size=time, name="Xs_ta")
        X_ancestors_ta = tf.TensorArray(tf.float32, size=time, name="X_ancestors_ta")
        log_Ws_ta = tf.TensorArray(tf.float32, size=time, name="log_Ws_ta")

        Xs_ta = Xs_ta.write(0, X_0)
        X_ancestors_ta = X_ancestors_ta.write(0, X_ancestor_0)
        log_Ws_ta = log_Ws_ta.write(0, log_W_0)
        preprocessed_obs_ta = \
            tf.TensorArray(tf.float32, size=time, name="preprocessed_obs_ta").unstack(self.preprocessed_obs)

        def while_cond(t, *unused_args):
            return t < time

        def while_body(t, X_ancestor_tm1, log_normalized_W_tm1, Xs_ta, X_ancestors_ta, log_Ws_ta):
            # q_f_t_feed = X_ancestor_tm1
            Input = tf.tile(tf.expand_dims(input[:, t - 1, :], axis=0),
                            (n_particles, 1, 1))  # (n_particles, batch_size, Dev)
            q_f_t_feed = tf.concat((X_ancestor_tm1, Input), axis=-1)  # (n_particles, batch_size, Dx + Dev)

            # proposal
            if self.q_uses_true_X:
                X_t, q_t_log_prob = self.sample_from_true_X(hidden[:, t, :],
                                                            q_cov,
                                                            sample_shape=(),
                                                            name="q_t_sample_and_log_prob")
                f_t_log_prob = f.log_prob(q_f_t_feed, X_t, name="f_t_log_prob")
            else:
                if self.model.use_2_q:
                    X_t, q_t_log_prob, f_t_log_prob = self.sample_from_2_dist(q1,
                                                                              self.q2,
                                                                              q_f_t_feed,
                                                                              preprocessed_obs_ta.read(t),
                                                                              inputs=Input,
                                                                              sample_size=())
                else:
                    X_t, q_t_log_prob = q1.sample_and_log_prob(q_f_t_feed,
                                                               sample_shape=(),
                                                               name="q_t_sample_and_log_prob")
                    # transition log probability
                    f_t_log_prob = f.log_prob(q_f_t_feed, X_t, name="f_t_log_prob", Dx=self.Dx)

            # emission log probability and log weights
            if self.log_dynamics or self.lar_dynamics:
                g_input = tf.exp(X_t)
            else:
                g_input = X_t
            if self.two_step_emission:
                g_input, _h_t_log_prob = self.h.sample_and_log_prob(g_input)
                if self.use_mask:
                    _h_t_log_prob_0 = tf.zeros_like(_h_t_log_prob)  # dummy values for missing observations
                    h_t_log_prob = tf.where(mask[0][0], _h_t_log_prob, _h_t_log_prob_0, name="h_t_log_prob")
                else:
                    h_t_log_prob = _h_t_log_prob
            _g_t_log_prob = self.g.log_prob(g_input, obs[:, t], extra_inputs=extra_inputs[:, t])
            if self.use_mask:
                _g_t_log_prob_0 = tf.zeros_like(_g_t_log_prob)
                g_t_log_prob = tf.where(mask[0][t], _g_t_log_prob, _g_t_log_prob_0, name="g_t_log_prob")
            else:
                g_t_log_prob = _g_t_log_prob

            log_alpha_t = tf.add(f_t_log_prob, g_t_log_prob - q_t_log_prob, name="log_alpha_t")
            if self.two_step_emission:
                log_alpha_t += h_t_log_prob - tf.stop_gradient(h_t_log_prob)

            log_W_t = log_alpha_t + log_normalized_W_tm1

            X_ancestor_t = self.resample_X(X_t, log_W_t, sample_size=n_particles,
                                           resample_particles=self.resample_particles)
            if self.resample_particles:
                log_normalized_W_t = -tf.log(tf.constant(n_particles, dtype=tf.float32, shape=(n_particles, batch_size)))
            else:
                log_normalized_W_t = log_W_t - tf.reduce_logsumexp(log_W_t, axis=0)

            # write results in this loop to tensor arrays
            Xs_ta = Xs_ta.write(t, X_t)
            X_ancestors_ta = X_ancestors_ta.write(t, X_ancestor_t)
            log_Ws_ta = log_Ws_ta.write(t, log_W_t)

            return t + 1, X_ancestor_t, log_normalized_W_t, Xs_ta, X_ancestors_ta, log_Ws_ta

        # conduct the while loop
        init_state = (1, X_ancestor_0, log_normalized_W_0, Xs_ta, X_ancestors_ta, log_Ws_ta)
        _, _, _, Xs_ta, X_ancestors_ta, log_Ws_ta = tf.while_loop(while_cond, while_body, init_state)

        # convert tensor arrays to tensors
        Xs = Xs_ta.stack()
        X_ancestors = X_ancestors_ta.stack()
        log_Ws = log_Ws_ta.stack()

        Xs.set_shape((None, n_particles, batch_size, Dx))
        X_ancestors.set_shape((None, n_particles, batch_size, Dx))
        log_Ws.set_shape((None, n_particles, batch_size))

        return Xs, X_ancestors, log_Ws

    def sample_from_2_dist(self, dist1, dist2, d1_input, d2_input, inputs=None, sample_size=()):
        d1_mvn = dist1.get_mvn(d1_input, Dx=self.Dx)
        d2_mvn = dist2.get_mvn(d2_input, Dx=self.Dx)

        if isinstance(d1_mvn, tfd.MultivariateNormalDiag) and isinstance(d2_mvn, tfd.MultivariateNormalDiag):
            if inputs is not None:
                for i in range(self.f_power-1):
                    d1_mvn_mean = d1_mvn.mean()
                    d1_mvn = dist1.get_mvn(tf.concat((d1_mvn_mean, inputs), axis=-1), Dx=self.Dx)

            d1_mvn_mean, d1_mvn_cov = d1_mvn.mean(), d1_mvn.stddev()
            d2_mvn_mean, d2_mvn_cov = d2_mvn.mean(), d2_mvn.stddev()

            d1_mvn_cov_inv, d2_mvn_cov_inv = 1 / d1_mvn_cov, 1 / d2_mvn_cov
            combined_cov = 1 / (d1_mvn_cov_inv + d2_mvn_cov_inv)
            combined_mean = combined_cov * (d1_mvn_cov_inv * d1_mvn_mean + d2_mvn_cov_inv * d2_mvn_mean)

            mvn = tfd.MultivariateNormalDiag(combined_mean,
                                             combined_cov,
                                             validate_args=True,
                                             allow_nan_stats=False)
        else:
            if isinstance(d1_mvn, tfd.MultivariateNormalDiag):
                d1_mvn_mean, d1_mvn_cov = d1_mvn.mean(), tf.diag(d1_mvn.stddev())
            elif isinstance(d1_mvn, tfd.MultivariateNormalFullCovariance):
                d1_mvn_mean, d1_mvn_cov = d1_mvn.mean(), d1_mvn.covariance()

            if isinstance(d2_mvn, tfd.MultivariateNormalDiag):
                d2_mvn_mean, d2_mvn_cov = d2_mvn.mean(), tf.diag(d2_mvn.stddev())
            elif isinstance(d2_mvn, tfd.MultivariateNormalFullCovariance):
                d2_mvn_mean, d2_mvn_cov = d2_mvn.mean(), d2_mvn.covariance()

            if len(d1_mvn_cov.shape.as_list()) == 2:
                d1_mvn_cov = tf.expand_dims(d1_mvn_cov, axis=0)

            d1_mvn_cov_inv, d2_mvn_cov_inv = tf.linalg.inv(d1_mvn_cov), tf.linalg.inv(d2_mvn_cov)
            combined_cov = tf.linalg.inv(d1_mvn_cov_inv + d2_mvn_cov_inv)
            perm = list(range(len(combined_cov.shape)))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            combined_cov = (combined_cov + tf.transpose(combined_cov, perm=perm)) / 2
            combined_mean = tf.matmul(combined_cov,
                                      tf.matmul(d1_mvn_cov_inv, tf.expand_dims(d1_mvn_mean, axis=-1)) +
                                      tf.matmul(d2_mvn_cov_inv, tf.expand_dims(d2_mvn_mean, axis=-1))
                                      )
            combined_mean = tf.squeeze(combined_mean, axis=-1)

            mvn = tfd.MultivariateNormalFullCovariance(combined_mean,
                                                       combined_cov,
                                                       validate_args=True,
                                                       allow_nan_stats=False)

        X = mvn.sample(sample_size)
        q_t_log_prob = mvn.log_prob(X)
        f_t_log_prob = d1_mvn.log_prob(X)

        return X, q_t_log_prob, f_t_log_prob

    def sample_from_true_X(self, hidden, q_cov, sample_shape=(), name="q_t_mvn"):
        mvn = tfd.MultivariateNormalDiag(hidden,
                                         q_cov * tf.ones(self.Dx, dtype=tf.float32),
                                         name=name)
        X = mvn.sample(sample_shape)
        q_t_log_prob = mvn.log_prob(X)

        return X, q_t_log_prob

    def resample_X(self, X, log_W, sample_size=(), resample_particles=True):
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
                if isinstance(X, list):
                    X_resampled = [tf.gather_nd(item, resample_idx) for item in X]
                else:
                    X_resampled = tf.gather_nd(X, resample_idx)
            else:
                assert sample_size == 1
                X_resampled = X
        else:
            X_resampled = X

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

            if not (self.model.use_bootstrap and self.model.use_2_q):
                preprocessed_X0 = self.model.X0_transformer(preprocessed_X0)

        return preprocessed_X0, preprocessed_obs

    def preprocess_obs_w_bRNN(self, obs, time_interval):
        self.y_smoother_f, self.y_smoother_b, self.X0_smoother_f, self.X0_smoother_b = self.model.bRNN
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

    @staticmethod
    def compute_MSE(y_hat_N_BxTxDy, y_N_BxTxDy, mask):
        MSE_ks = []  # [MSE_0, MSE_1, ..., MSE_N]
        y_means = []  # [y_mean_0 (shape = Dy), ..., y_mean_N], used to calculate y_var across all batches
        y_vars = []  # [y_var_0 (shape = Dy), ..., y_var_N], used to calculate y_var across all batches
        for k, (y_hat_BxTmkxDy, y_BxTmkxDy) in enumerate(zip(y_hat_N_BxTxDy, y_N_BxTxDy)):
            difference = y_hat_BxTmkxDy - y_BxTmkxDy  # (batch_size, time-k, Dy)
            masked_difference = tf.boolean_mask(difference, mask[:, k:])  # (time-k, Dy)
            masked_difference = masked_difference[None,]  # (batch_size, time-k, Dy)

            MSE_k = tf.reduce_sum(masked_difference ** 2, name="MSE_{}".format(k))
            MSE_ks.append(MSE_k)

            masked_y = tf.boolean_mask(y_BxTmkxDy, mask[:, k:])  # (mask_time, Dy)
            masked_y = masked_y[None,]  # (batch_size, mask_time, Dy)
            y_mean = tf.reduce_mean(masked_y, axis=[0, 1], name="y_mean_{}".format(k))  # (Dy,)
            y_means.append(y_mean)
            y_var = tf.reduce_sum((masked_y - y_mean) ** 2, axis=[0, 1], name="y_var_{}".format(k))  # (Dy,)
            y_vars.append(y_var)

        MSE_ks = tf.stack(MSE_ks, name="MSE_ks")  # (n_steps + 1)
        y_means = tf.stack(y_means, name="y_means")  # (n_steps + 1, Dy)
        y_vars = tf.stack(y_vars, name="y_vars")  # (n_steps + 1, Dy)

        y_hat_N_BxTxDy = list(y_hat_N_BxTxDy)  # deep copy
        return MSE_ks, y_means, y_vars, y_hat_N_BxTxDy

    def n_step_MSE(self, n_steps, X, hidden, obs, input, mask, extra_inputs):
        """
        Compute MSE_k for k = 0, ..., n_steps. This is an intermediate step to calculate k-step R^2
        :param n_steps: integer
        :param hidden: (batch_size, time, n_particles, Dx)
        :param obs: (batch_size, time, Dy)
        :param input: (batch_size, time, Dv)
        :param mask: (batch_size, time)
        :param extra_inputs: (batch_size, time)
        :return:
        """

        batch_size, _, _, _ = X.shape.as_list()

        assert batch_size == 1

        _, _, Dy = obs.shape.as_list()  # (batch_size, time, Dy)
        # assert n_steps < time, "n_steps = {} >= time".format(n_steps)

        with tf.variable_scope(self.name):
            X = tf.reduce_mean(X, axis=2)  # average over paths (n_particles),shape (batch_size, time, Dx)
            x_BxTmkxDz = X

            # get y_hat
            y_hat_N_BxTxDy = []
            y_hat_unmasked_N_BxTxDy = []

            for k in range(n_steps):
                if self.log_dynamics or self.lar_dynamics:
                    g_input = tf.exp(x_BxTmkxDz)
                else:
                    g_input = x_BxTmkxDz
                if self.two_step_emission:
                    g_input = self.h.mean(g_input)
                y_hat_BxTmkxDy = self.g.mean(g_input, extra_inputs=extra_inputs[:, k:])
                # (batch_size, time - k, Dy)
                y_hat_unmasked_N_BxTxDy.append(y_hat_BxTmkxDy)
                y_hat_BxTmkxDy = tf.boolean_mask(y_hat_BxTmkxDy, mask[:, k:])[tf.newaxis, :, :]
                y_hat_N_BxTxDy.append(y_hat_BxTmkxDy)

                x_BxTmkxDz = x_BxTmkxDz[:, :-1]  # (batch_size, time - k - 1, Dx)

                f_k_feed = tf.concat([x_BxTmkxDz, input[:, k:-1]], axis=-1)         # (batch_size, time - k - 1, Dx+Dev)
                f_k_feed = tf.transpose(f_k_feed, [1, 0, 2])
                x_BxTmkxDz = self.f.mean(f_k_feed, Dx=self.Dx)   # (batch_size, time - k - 1, Dx)
                x_BxTmkxDz = tf.transpose(x_BxTmkxDz, [1, 0, 2])

            if self.log_dynamics or self.lar_dynamics:
                g_input = tf.exp(x_BxTmkxDz)
            else:
                g_input = x_BxTmkxDz
            if self.two_step_emission:
                g_input = self.h.mean(g_input)
            y_hat_BxTmNxDy = self.g.mean(g_input, extra_inputs=extra_inputs[:, n_steps:])   # (batch_size, T - N, Dy)
            y_hat_unmasked_N_BxTxDy.append(y_hat_BxTmkxDy)
            y_hat_BxTmNxDy = tf.boolean_mask(y_hat_BxTmNxDy, mask[:, n_steps:])[tf.newaxis, :, :]
            y_hat_N_BxTxDy.append(y_hat_BxTmNxDy)

            # get y_true
            y_N_BxTxDy = []
            for k in range(n_steps + 1):
                y_BxTmkxDy = obs[:, k:, :]
                y_BxTmkxDy = tf.boolean_mask(y_BxTmkxDy, mask[:, k:])[tf.newaxis, :, :]
                y_N_BxTxDy.append(y_BxTmkxDy)

        return y_hat_N_BxTxDy, y_N_BxTxDy, y_hat_unmasked_N_BxTxDy

    def get_nextX(self, X):
        # only used for drawing 2D quiver plot
        with tf.variable_scope(self.name):
            return self.f.mean(X, Dx=self.Dx)

    @staticmethod
    def lar_transform(percentages):
        """

        :param percentages: (..., dy),
        :return: lars: (..., dy - 1)
        """
        lars = tf.log(percentages[..., :-1]) - tf.log(percentages[..., -1:])
        return lars

