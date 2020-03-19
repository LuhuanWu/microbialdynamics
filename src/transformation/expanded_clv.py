import tensorflow as tf

from src.transformation.base import transformation

"""
transforming the beta distribution
"""

EPSILON = 1e-8


class ExpandedCLVTransformation(transformation):
    def __init__(self, Dx, Dev, Dy, clv_in_alr=True, training=True, clip_alpha=8., threshold=3.):
        self.Dx = Dx
        self.Dev = Dev
        self.Dy = Dy
        self.clv_in_alr = clv_in_alr
        self.training = training
        self.clip_alpha = clip_alpha
        self.threshold = threshold

        # batch_matrices for beta. beta is (n_topis, n_words) = (Dx+1, Dy)
        # for each topic, there is a set of matrix
        if clv_in_alr:
            self.A_var = tf.Variable(tf.zeros((self.Dx + 1, self.Dy, self.Dy)))
            self.g_var = tf.Variable(tf.zeros((self.Dx + 1, self.Dy)))
            self.Wg_var = tf.Variable(tf.zeros((self.Dx + 1, self.Dev, self.Dy)))
        else:
            self.A_var = tf.Variable(tf.zeros((self.Dx, self.Dy, self.Dy)))
            self.g_var = tf.Variable(tf.zeros((self.Dx, self.Dy)))
            self.Wg_var = tf.Variable(tf.zeros((self.Dx, self.Dev, self.Dy)))

        self.A_beta = tf.nn.softplus(self.A_var)                          # off-diagonal elements should be positive
        self_interaction = tf.linalg.diag_part(self.A_beta)
        self.A_beta = tf.linalg.set_diag(self.A_beta, -self_interaction)  # self-interaction should be negative
        self.g_beta = tf.nn.softplus(self.g_var)                          # growth should be positive
        self.Wg_beta = self.Wg_var

        self.A_beta_log_sigma2 = tf.Variable(-10 * tf.ones((self.Dx, self.Dy, self.Dy)))
        log_alpha = compute_log_alpha(self.A_beta_log_sigma2, self.A_beta, EPSILON, value_limit=None)
        weight_mask = tf.cast(tf.less(log_alpha, threshold), tf.float32)
        self.dropout_A_beta = self.A_beta * weight_mask

    def transform(self, Input):
        """
        :param Input: [v, beta_log].  v(batch_shape_v), beta_log (batch_shape_beta, Dx+1, Dy)
        batch_shape_v should match the trailing shape of batch_shape_beta
        :return: output: (..., batch_size, Dx+1, Dy) if self.clv_in_alr else (..., batch_size, Dx, Dy)
        """
        # x_t + g_t + v_t * Wg + p_t * A
        assert isinstance(Input, list), type(Input)
        assert len(Input) == 2, len(Input)
        v, beta_log = Input
        assert v.shape.as_list()[-1] == self.Dev, "v_dim = {}, Dev = {}".format(v.shape.as_list()[-1], self.Dev)

       # check batch shape compatibility
        batch_shape_v, batch_shape_beta_log = v.shape[:-1], beta_log.shape[:-2]
        assert len(batch_shape_v) <= len(batch_shape_beta_log), "{}, {}".format(batch_shape_v, batch_shape_beta_log)

        if self.clv_in_alr:
            zeros = tf.zeros_like(beta_log[..., 0:1])
            beta_log = tf.concat([beta_log, zeros], axis=-1)
        p_beta = tf.nn.softmax(beta_log, axis=-1)  # (..., Dx, Dy) if self.clv_in_alr else (..., Dx, Dy)

        pA = tf.cond(self.training,
                     lambda: matmul_train(p_beta, (self.A_beta, self.A_beta_log_sigma2), clip_alpha=self.clip_alpha),
                     lambda: matmul_eval(p_beta, (self.A_beta, self.A_beta_log_sigma2), threshold=self.threshold))
        # (..., Dx + 1, Dy, 1) * (Dx + 1, Dy, Dy) if self.clv_in_alr else (..., Dx, Dy, 1) * (Dx, Dy, Dy)
        # pA = tf.reduce_sum(p_beta[..., None] * self.A_beta, axis=-2)  # (...,  Dx + 1, Dy) or (...,  Dx, Dy)
        if self.Dev > 0:
            assert v.shape[-1] == self.Dev

            v = tf.expand_dims(v, axis=-2) # (..., 1, Dev)
            # (..., 1, Dev, 1) * (Dx + 1, Dev, Dy)
            Wg_beta_v = tf.reduce_sum(v[..., None] * self.Wg_beta, axis=-2) # (..., Dx + 1, Dy)

            output_beta_log = beta_log + self.g_beta + Wg_beta_v + pA
        else:
            output_beta_log = beta_log + self.g_beta + pA

        if self.clv_in_alr:
            output_beta_log = output_beta_log[..., :-1] - output_beta_log[..., -1:]
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