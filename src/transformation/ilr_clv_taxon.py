import numpy as np
import tensorflow as tf

from src.transformation.base import transformation
from src.transformation.ilr_clv_utils import convert_theta_to_tree
from src.transformation.ilr_clv_utils import setup_heights, get_inode_and_taxon_idxes
from src.transformation.ilr_clv_utils import get_psi, inverse_ilr_transform, get_inode_relative_abundance
from src.transformation.l0_utils import hard_concrete_sample, hard_concrete_mean, l0_norm, GAMMA, ZETA, BETA

EPS = 1e-6


class ilr_clv_taxon_transformation(transformation):
    def __init__(self, theta, Dv,
                 reg_coef=1.0, params_reg_func="L2", annealing_frac=1.0):
        self.theta = theta
        Dx, Dy = self.Dx, self.Dy = theta.shape
        self.Dv = Dv

        assert params_reg_func in ["L1", "L2"]
        self.params_reg_func = params_reg_func
        self.reg_coef = reg_coef
        self.annealing_frac = annealing_frac

        # init tree
        self.root, self.reference = convert_theta_to_tree(self.theta)
        setup_heights(self.root)
        self.psi = get_psi(self.theta)

        # init variable
        self.A = tf.Variable(tf.zeros((Dx, Dy)))
        self.g = tf.Variable(tf.zeros((Dx,)))
        self.Wv = tf.Variable(tf.zeros((Dx, self.Dv)))

        self.params = {"A": self.A, "g": self.g, "Wv": self.Wv}

    def regularization_loss(self):

        if self.params_reg_func == "L1":
            params_reg = tf.reduce_sum(tf.abs(self.A)) + tf.reduce_sum(tf.abs(self.g)) + tf.reduce_sum(tf.abs(self.Wv))
        else:  # L2
            params_reg = tf.reduce_sum(self.A ** 2) + tf.reduce_sum(self.g ** 2) + tf.reduce_sum(self.Wv ** 2)

        with tf.variable_scope('reg_loss'):
            tf.summary.scalar('params_reg', params_reg * self.reg_coef)

        return params_reg * self.reg_coef * self.annealing_frac

    def transform(self, Input):
        """
        :param Input: (n_particles, batch_size, Dx + Dv)
        :param Dx: dimension of hidden space
        :return: output: (n_particles, batch_size, Dx)
        """
        # x_t + g_t + v_t * Wv + p_t * A

        A, g, Wv = tf.transpose(self.A, [1, 0]), self.g, tf.transpose(self.Wv, [1, 0])
        Dx = self.Dx

        x = Input[..., :Dx]  # (n_particles, batch_size, Dx)
        v = Input[0, 0:1, Dx:]  # (1, Dv)
        v_size = v.shape[-1]

        p_t = inverse_ilr_transform(x, self.psi)

        # (..., Dx)
        pA = tf.reduce_sum(p_t[..., None] * A, axis=-2)

        delta = g + pA
        if v_size > 0:
            # Wv shape (Dv, Dx)
            Wvv = tf.reduce_sum(v[..., None] * Wv, axis=-2)  # (..., Dx)
            delta += Wvv

        output = x + delta

        return output
