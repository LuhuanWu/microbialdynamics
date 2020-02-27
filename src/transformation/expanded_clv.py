import tensorflow as tf

from src.transformation.base import transformation

"""
transforming the beta distribution
"""
class ExpandedCLVTransformation(transformation):
    def __init__(self, Dx, Dev, Dy, clv_in_alr=True):
        self.Dx = Dx
        self.Dev = Dev
        self.Dy = Dy
        self.clv_in_alr = clv_in_alr

        # batch_matrices for beta. beta is (n_topis, n_words) = (Dx+1, Dy)
        # for each topic, there is a set of matrix
        if clv_in_alr:
            self.A_var = tf.Variable(tf.zeros((self.Dx + 1, self.Dy, self.Dy)))
            self.g_var = tf.Variable(tf.zeros((1, self.Dy)))
            self.Wg_var = tf.Variable(tf.zeros((1, self.Dev, self.Dy)))
        else:
            self.A_var = tf.Variable(tf.zeros((self.Dx, self.Dy, self.Dy)))
            self.g_var = tf.Variable(tf.zeros((1, self.Dy)))
            self.Wg_var = tf.Variable(tf.zeros((1, self.Dev, self.Dy)))

        self.A_beta = tf.nn.softplus(self.A_var)                          # off-diagonal elements should be positive
        self_interaction = tf.linalg.diag_part(self.A_beta)
        self.A_beta = tf.linalg.set_diag(self.A_beta, -self_interaction)  # self-interaction should be negative
        self.g_beta = tf.nn.softplus(self.g_var)                          # growth should be positive
        self.Wg_beta = self.Wg_var

    def transform(self, Input):
        """
        :param Input: [v, beta_log].  v(batch_shape_v), beta_log (batch_shape_beta, Dx+1, Dy-1)
        batch_shape_v should match the trailing shape of batch_shape_beta
        :return: output: (..., batch_size, Dx+1, Dy-1)
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
        p_beta = tf.nn.softmax(beta_log, axis=-1)  # (..., Dx + 1, Dy)

        # (..., Dx+1, Dy, 1) * (Dx+1, Dy, Dy-1)
        pA = tf.reduce_sum(p_beta[..., None] * self.A_beta, axis=-2) # (...,  Dx+1, Dy-1)
        if self.Dev > 0:
            assert v.shape[-1] == self.Dev

            v = tf.expand_dims(v, axis=-2) # (..., 1, Dev)
            # (..., 1, Dev, 1) * (Dx+1, Dev, Dy-1)
            Wg_beta_v = tf.reduce_sum(v[..., None] * self.Wg_beta, axis=-2) # (..., Dx+1, Dy-1)

            output_beta_log = beta_log + self.g_beta + Wg_beta_v + pA
        else:
            output_beta_log = beta_log + self.g_beta + pA

        if self.clv_in_alr:
            output_beta_log = output_beta_log[..., :-1] - output_beta_log[..., -1:]
        return output_beta_log

