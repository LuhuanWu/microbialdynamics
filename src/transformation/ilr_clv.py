import tensorflow as tf

from src.transformation.base import transformation
from src.transformation.ilr_clv_utils import convert_theta_to_tree, get_p_i, get_p_j_given_i, get_psi
from src.transformation.ilr_clv_utils import inverse_ilr_transform, get_inode_relative_abundance
from src.transformation.l0_utils import hard_concrete_sample, hard_concrete_mean, l0_norm

EPS = 1e-6


class ilr_clv_transformation(transformation):
    def __init__(self, theta, Dev, training=False, b_dropout_rate=0.5):
        self.Dx, self.Dy = theta.shape
        self.Dev = Dev

        # init variable
        A_init_val = tf.zeros((self.Dx, self.Dx + self.Dy))
        g_init_val = tf.zeros((self.Dx,))
        Wv_init_val = tf.zeros((self.Dx, self.Dev))
        self.A_var = tf.Variable(A_init_val)
        self.g_var = tf.Variable(g_init_val)
        self.Wv_var = tf.Variable(Wv_init_val)

        self.A = self.A_var
        self.g = self.g_var
        self.Wv = self.Wv_var

        # init tree
        self.root, self.reference = convert_theta_to_tree(theta)
        self.psi = get_psi(theta)

        # L0 regularization for break score
        log_alpha_init = tf.log(b_dropout_rate / (1 - b_dropout_rate))
        self.log_alpha = tf.Variable(log_alpha_init * tf.ones(self.Dx, dtype=tf.float32))
        b_noises = tf.cond(training,
                           lambda: hard_concrete_sample(self.log_alpha),
                           lambda: hard_concrete_mean(self.log_alpha))
        b = []
        b_before_gated = []
        for inode, b_noise in zip(self.reference[:self.Dx], tf.unstack(b_noises)):
            b_before_gated.append(inode.b)
            inode.b = inode.b * b_noise
            b.append(inode.b)
        self.b_before_gated = tf.stack(b_before_gated)
        self.b = tf.stack(b)

        self.p_i = get_p_i(self.Dx, self.root)
        self.p_j_given_i = get_p_j_given_i(self.Dx + self.Dy, self.Dx, self.root)

        self.A *= self.p_i[:, None] * self.p_j_given_i
        self.g *= self.p_i
        self.Wv *= self.p_i[:, None]

        self.params = {"b": self.b, "A": self.A, "g": self.g, "Wv": self.Wv,
                       "p_i": self.p_i, "p_j_given_i": self.p_j_given_i}

    def regularization_loss(self):
        L0 = l0_norm(self.log_alpha)
        L2 = tf.reduce_sum(self.A ** 2) + tf.reduce_sum(self.g ** 2) + tf.reduce_sum(self.Wv ** 2)
        b_regularization = tf.log(1 - (2 * self.b_before_gated - 1) ** 2 + EPS)
        return L0 + L2 + b_regularization

    def transform(self, Input):
        """
        :param Input: (n_particles, batch_size, Dx + Dev)
        :param Dx: dimension of hidden space
        :return: output: (n_particles, batch_size, Dx)
        """
        # x_t + g_t + v_t * Wv + p_t * A

        A, g, Wv = tf.transpose(self.A, [1, 0]), self.g, tf.transpose(self.Wv, [1, 0])
        Dx = self.Dx

        x = Input[..., :Dx]  # (n_particles, batch_size, Dx)
        v = Input[0, 0:1, Dx:]  # (1, Dev)
        v_size = v.shape[-1]

        p_t = inverse_ilr_transform(x, self.psi)
        r_t = get_inode_relative_abundance(self.root, p_t, Dx)
        p_t = tf.concat([r_t, p_t], axis=-1)

        # (..., Dx + Dy, 1) * (Dx + Dy, Dx)
        pA = tf.reduce_sum(p_t[..., None] * A, axis=-2)  # (..., Dx)

        delta = g + pA
        if v_size > 0:
            # Wv shape (Dev, Dx)
            Wvv = tf.reduce_sum(v[..., None] * Wv, axis=-2)  # (n_particles, batch_size, Dx)
            delta += Wvv

        output = x + delta

        return output
