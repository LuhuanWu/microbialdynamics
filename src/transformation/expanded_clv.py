import pickle
import numpy as np
import tensorflow as tf

from src.transformation.base import transformation

"""
transforming the beta distribution
"""

EPSILON = 1e-8


class ExpandedCLVTransformation(transformation):
    def __init__(self, Dx, Dev, Dy, training=True,
                 use_variational_dropout=False, clip_alpha=8., threshold=3.,
                 use_anchor=False, anchor_x=[],
                 data_dir=None):
        self.Dx = Dx
        self.Dev = Dev
        self.Dy = Dy
        self.training = training
        self.clip_alpha = clip_alpha
        self.threshold = threshold
        self.use_anchor = use_anchor
        self.anchor_x = anchor_x
        if use_anchor:
            assert len(anchor_x) > 0

        # batch_matrices for beta. beta is (n_topis, n_words) = (Dx+1, Dy)
        # for each topic, there is a set of matrix
        if data_dir is not None:
            with open(data_dir, "rb") as f:
                d = pickle.load(f)
            A_init_val = tf.constant(d['A_g'], dtype=tf.float32)
            g_g = np.array([d['g_g']] * self.Dx)
            Wv_g = np.array([d['Wv_g'].T] * self.Dx)
            g_init_val = tf.constant(g_g, dtype=tf.float32)
            Wv_init_val = tf.constant(Wv_g, dtype=tf.float32)
        else:
            A_init_val = tf.zeros((self.Dx, self.Dy, self.Dy))
            g_init_val = tf.zeros((self.Dx, self.Dy))
            Wv_init_val = tf.zeros((self.Dx, self.Dev, self.Dy))
        self.A_var = tf.Variable(A_init_val)
        self.g_var = tf.Variable(g_init_val)
        self.Wv_var = tf.Variable(Wv_init_val)

        self.A_beta = tf.nn.softplus(self.A_var)                          # off-diagonal elements should be positive
        self_interaction = tf.linalg.diag_part(self.A_beta)
        self.A_beta = tf.linalg.set_diag(self.A_beta, -self_interaction)  # self-interaction should be negative
        self.g_beta = tf.nn.softplus(self.g_var)                          # growth should be positive
        self.Wv_beta = self.Wv_var

        if use_variational_dropout:
            self.A_beta_log_sigma2 = tf.Variable(-10 * tf.ones((self.Dx, self.Dy, self.Dy)))
        else:
            self.A_beta_log_sigma2 = -10 * tf.ones((self.Dx, self.Dy, self.Dy), dtype=tf.float32)
        log_alpha = compute_log_alpha(self.A_beta_log_sigma2, self.A_beta, EPSILON, value_limit=None)
        weight_mask = tf.cast(tf.less(log_alpha, threshold), tf.float32)
        self.dropout_A_beta = self.A_beta * weight_mask

    def transform(self, Input):
        """
        :param Input: [v, beta_log].  v(batch_shape_v), beta_log (batch_shape_beta, Dx+1, Dy)
        batch_shape_v should match the trailing shape of batch_shape_beta
        :return: output: (..., batch_size, Dx, Dy)
        """
        # x_t + g_t + v_t * Wv + p_t * A
        assert isinstance(Input, list), type(Input)
        assert len(Input) == 2, len(Input)
        v, beta_log = Input
        assert v.shape.as_list()[-1] == self.Dev, "v_dim = {}, Dev = {}".format(v.shape.as_list()[-1], self.Dev)

        # check batch shape compatibility
        batch_shape_v, batch_shape_beta_log = v.shape[:-1], beta_log.shape[:-2]
        assert len(batch_shape_v) <= len(batch_shape_beta_log), "{}, {}".format(batch_shape_v, batch_shape_beta_log)

        beta_log_ = beta_log

        if self.use_anchor:
            ones = tf.ones_like(beta_log_[..., 0:1])
            anchors = [ones * x_val for x_val in self.anchor_x]
            beta_log_ = tf.concat([beta_log_] + anchors, axis=-1)

        p_beta = tf.nn.softmax(beta_log_, axis=-1)  # (n_particles, batch_size, Dx + 1)

        if self.use_anchor:
            p_beta = p_beta[..., :-len(self.anchor_x)]

        pA = tf.cond(self.training,
                     lambda: matmul_train(p_beta, (self.A_beta, self.A_beta_log_sigma2), clip_alpha=self.clip_alpha),
                     lambda: matmul_eval(p_beta, (self.A_beta, self.A_beta_log_sigma2), threshold=self.threshold))
        # (..., Dx, Dy, 1) * (Dx, Dy, Dy)
        # pA = tf.reduce_sum(p_beta[..., None] * self.A_beta, axis=-2)  # (...,  Dx + 1, Dy) or (...,  Dx, Dy)
        if self.Dev > 0:
            assert v.shape[-1] == self.Dev

            v = tf.expand_dims(v, axis=-2) # (..., 1, Dev)
            # (..., 1, Dev, 1) * (Dx + 1, Dev, Dy)
            Wv_beta_v = tf.reduce_sum(v[..., None] * self.Wv_beta, axis=-2) # (..., Dx + 1, Dy)

            output_beta_log = beta_log + self.g_beta + Wv_beta_v + pA
        else:
            output_beta_log = beta_log + self.g_beta + pA

        return output_beta_log


    def variational_dropout_dkl_loss(self):
        log_alpha = compute_log_alpha(self.A_beta_log_sigma2, self.A_beta, EPSILON, self.clip_alpha)

        # Constant values for approximating the kl divergence
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        c = -k1

        # Compute each term of the KL and combine
        term_1 = k1 * tf.nn.sigmoid(k2 + k3 * log_alpha)
        term_2 = -0.5 * tf.log1p(tf.exp(tf.negative(log_alpha)))
        eltwise_dkl = term_1 + term_2 + c
        return -tf.reduce_sum(eltwise_dkl)


# https://github.com/google-research/google-research/blob/master/state_of_sparsity/layers/variational_dropout/common.py
def compute_log_alpha(log_sigma2, theta, eps=EPSILON, value_limit=8.):
  R"""Compute the log \alpha values from \theta and log \sigma^2.
  The relationship between \sigma^2, \theta, and \alpha as defined in the
  paper https://arxiv.org/abs/1701.05369 is
  \sigma^2 = \alpha * \theta^2
  This method calculates the log \alpha values based on this relation.
  Args:
    log_sigma2: tf.Variable. The log variance for each weight.
    theta: tf.Variable. The mean for each weight.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.
    value_limit: If not None, the log_alpha values will be clipped to the
     range [-value_limit, value_limit]. This is consistent with the
     implementation provided with the publication.
  Returns:
    A tf.Tensor representing the calculated log \alpha values.
  """
  log_alpha = log_sigma2 - tf.log(tf.square(theta) + eps)

  if value_limit is not None:
    # If a limit is specified, clip the alpha values
    return tf.clip_by_value(log_alpha, -value_limit, value_limit)
  return log_alpha


def compute_log_sigma2(log_alpha, theta, eps=EPSILON):
  R"""Compute the log \sigma^2 values from log \alpha and \theta.
  The relationship between \sigma^2, \theta, and \alpha as defined in the
  paper https://arxiv.org/abs/1701.05369 is
  \sigma^2 = \alpha * \theta^2
  This method calculates the log \sigma^2 values based on this relation.
  Args:
    log_alpha: tf.Tensor. The log alpha values for each weight.
    theta: tf.Variable. The mean for each weight.
    eps: Small constant value to use in log and sqrt operations to avoid NaNs.
  Returns:
    A tf.Tensor representing the calculated log \sigma^2 values.
  """
  return log_alpha + tf.log(tf.square(theta) + eps)

def matmul_train(x, variational_params, clip_alpha=None, eps=EPSILON):
    theta, log_sigma2 = variational_params

    if clip_alpha is not None:
        # Compute the log_alphas and then compute the
        # log_sigma2 again so that we can clip on the
        # log alpha magnitudes
        log_alpha = compute_log_alpha(log_sigma2, theta, eps, clip_alpha)
        log_sigma2 = compute_log_sigma2(log_alpha, theta, eps)

    # Compute the mean and standard deviation of the distributions
    x = x[..., None]
    mu = tf.reduce_sum(x * theta, axis=-2)
    std = tf.sqrt(tf.reduce_sum(tf.square(x) * tf.exp(log_sigma2), axis=-2) + eps)

    output_shape = tf.shape(std)
    return mu + std * tf.random_normal(output_shape)

def matmul_eval(x, variational_params, threshold=None, eps=EPSILON):
    theta, log_sigma2 = variational_params

    # Compute the weight mask by thresholding on
    # the log-space alpha values
    log_alpha = compute_log_alpha(log_sigma2, theta, eps, value_limit=None)
    weight_mask = tf.cast(tf.less(log_alpha, threshold), tf.float32)

    theta *= weight_mask
    x = x[..., None]
    mu = tf.reduce_sum(x * theta, axis=-2)
    return mu