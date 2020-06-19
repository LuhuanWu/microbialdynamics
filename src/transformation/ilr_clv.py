import numpy as np
import tensorflow as tf

from src.transformation.base import transformation
from src.transformation.ilr_clv_utils import convert_theta_to_tree
from src.transformation.ilr_clv_utils import get_p_i, get_between_group_p_j_given_i, get_psi, get_in_group_p_j_given_i
from src.transformation.ilr_clv_utils import inverse_ilr_transform, get_inode_relative_abundance
from src.transformation.l0_utils import hard_concrete_sample, hard_concrete_mean, l0_norm

EPS = 1e-6


class ilr_clv_transformation(transformation):
    def __init__(self, theta, Dev,
                 exist_in_group_dynamics=False, training=False,
                 use_L0=True, L0_reg_coef=1.0, b_dropout_rate=0.9, annealing_frac=1.0):
        self.Dx, self.Dy = theta.shape
        self.Dev = Dev
        self.use_L0 = use_L0
        self.L0_reg_coef = L0_reg_coef
        self.annealing_frac = annealing_frac

        # init variable
        self.A_in_var = tf.Variable(tf.zeros((self.Dx, self.Dx + self.Dy)))
        self.g_in_var = tf.Variable(tf.zeros((self.Dx,)))
        self.Wv_in_var = tf.Variable(tf.zeros((self.Dx, self.Dev)))

        self.A_between_var = tf.Variable(tf.zeros((self.Dx, self.Dx + self.Dy)))
        self.g_between_var = tf.Variable(tf.zeros((self.Dx,)))
        self.Wv_between_var = tf.Variable(tf.zeros((self.Dx, self.Dev)))

        # init tree
        self.root, self.reference = convert_theta_to_tree(theta)
        self.psi = get_psi(theta)

        inode_depth = []
        for inode in self.reference[:self.Dx]:
            inode_depth.append(inode.depth)
        training_mask = (np.array(inode_depth) / (np.max(inode_depth) + 1.0)) < annealing_frac

        # L0 regularization for break score
        if use_L0:
            log_alpha_init = tf.log(b_dropout_rate / (1 - b_dropout_rate))
            self.log_alpha = tf.Variable(log_alpha_init * tf.ones(self.Dx, dtype=tf.float32))
            self.log_alpha = tf.where(training_mask, self.log_alpha, tf.stop_gradient(self.log_alpha))
            b_noises = tf.cond(training,
                               lambda: hard_concrete_sample(self.log_alpha),
                               lambda: hard_concrete_mean(self.log_alpha))
        else:
            b_noises = tf.ones(self.Dx, dtype=tf.float32)

        b = []
        b_before_gated = []
        self.A_between_var_list = tf.unstack(self.A_between_var, axis=-1)
        self.A_between_var_masked = [0 for _ in range(self.Dx + self.Dy)]
        for inode, b_noise, mask in zip(self.reference[:self.Dx], tf.unstack(b_noises), tf.unstack(training_mask)):
            b_before_gated.append(inode.b)
            inode.b = tf.where(mask, inode.b, tf.stop_gradient(inode.b))
            inode.b = inode.b * b_noise
            b.append(inode.b)

            idxes = [inode.left.node_idx, inode.right.node_idx]
            if inode == self.root:
                idxes.append(inode.node_idx)
            for idx in idxes:
                node_A_var = self.A_between_var_list[idx]
                self.A_between_var_masked[idx] = tf.where(mask, node_A_var, tf.stop_gradient(node_A_var))

        self.b_before_gated = tf.stack(b_before_gated)
        self.b = tf.stack(b)

        self.A_between_var = tf.stack(self.A_between_var_masked, axis=-1)
        self.g_between_var = tf.where(training_mask, self.g_between_var, tf.stop_gradient(self.g_between_var))
        self.Wv_between_var = tf.where(training_mask, self.Wv_between_var, tf.stop_gradient(self.Wv_between_var))

        self.p_i = get_p_i(self.Dx, self.root)
        self.in_group_p_j_given_i = get_in_group_p_j_given_i(self.Dx + self.Dy, self.Dx, self.root)
        self.between_group_p_j_given_i = get_between_group_p_j_given_i(self.Dx + self.Dy, self.Dx, self.root)

        self.A_in = self.A_in_var * self.in_group_p_j_given_i * exist_in_group_dynamics
        self.g_in = self.g_in_var * (1.0 - self.p_i) * exist_in_group_dynamics
        self.Wv_in = self.Wv_in_var * (1.0 - self.p_i[:, None]) * exist_in_group_dynamics

        self.A_between = self.A_between_var * self.p_i[:, None] * self.between_group_p_j_given_i
        self.g_between = self.g_between_var * self.p_i
        self.Wv_between = self.Wv_between_var * self.p_i[:, None]

        self.params = {"b": self.b,
                       "A_in": self.A_in, "g_in": self.g_in, "Wv_in": self.Wv_in,
                       "A_between": self.A_between, "g_between": self.g_between, "Wv_between": self.Wv_between,
                       "p_i": self.p_i,
                       "in_group_p_j_given_i": self.in_group_p_j_given_i,
                       "between_group_p_j_given_i": self.between_group_p_j_given_i}

    def regularization_loss(self):
        if self.use_L0:
            L0 = l0_norm(self.log_alpha)
        else:
            L0 = 0.0
        L2 = tf.reduce_sum(self.A_in_var ** 2) + tf.reduce_sum(self.g_in_var ** 2) + \
             tf.reduce_sum(self.Wv_in_var ** 2) + \
             tf.reduce_sum(self.A_between_var ** 2) + tf.reduce_sum(self.g_between_var ** 2) + \
             tf.reduce_sum(self.Wv_between_var ** 2)
        L1 = tf.reduce_sum(tf.abs(self.A_in_var)) + tf.reduce_sum(tf.abs(self.g_in_var)) + \
             tf.reduce_sum(tf.abs(self.Wv_in_var)) + \
             tf.reduce_sum(tf.abs(self.A_between_var)) + tf.reduce_sum(tf.abs(self.g_between_var)) + \
             tf.reduce_sum(tf.abs(self.Wv_between_var))
        b_regularization = tf.log(1 - tf.abs(self.b_before_gated - 0.5) ** 3 + EPS)
        return (L0 + L2 + b_regularization) * self.L0_reg_coef  * self.annealing_frac

    def transform(self, Input):
        """
        :param Input: (n_particles, batch_size, Dx + Dev)
        :param Dx: dimension of hidden space
        :return: output: (n_particles, batch_size, Dx)
        """
        # x_t + g_t + v_t * Wv + p_t * A

        A_in, g_in, Wv_in = tf.transpose(self.A_in, [1, 0]), self.g_in, tf.transpose(self.Wv_in, [1, 0])
        A_between, g_between, Wv_between = \
            tf.transpose(self.A_between, [1, 0]), self.g_between, tf.transpose(self.Wv_between, [1, 0])
        Dx = self.Dx

        x = Input[..., :Dx]  # (n_particles, batch_size, Dx)
        v = Input[0, 0:1, Dx:]  # (1, Dev)
        v_size = v.shape[-1]

        p_t = inverse_ilr_transform(x, self.psi)
        r_t = get_inode_relative_abundance(self.root, p_t, Dx)
        p_t = tf.concat([r_t, p_t], axis=-1)

        # (..., Dx + Dy, 1) * (Dx + Dy, Dx)
        pA = tf.reduce_sum(p_t[..., None] * (A_in + A_between), axis=-2)  # (..., Dx)

        delta = g_in + g_between + pA
        if v_size > 0:
            # Wv shape (Dev, Dx)
            Wvv = tf.reduce_sum(v[..., None] * (Wv_in + Wv_between), axis=-2)  # (n_particles, batch_size, Dx)
            delta += Wvv

        output = x + delta

        return output