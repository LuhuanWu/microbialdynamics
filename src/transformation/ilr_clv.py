import numpy as np
import tensorflow as tf

from src.transformation.base import transformation
from src.transformation.ilr_clv_utils import convert_theta_to_tree
from src.transformation.ilr_clv_utils import get_inode_depths, get_inode_heights, get_inode_and_taxon_idxes
from src.transformation.ilr_clv_utils import get_p_i, get_between_group_p_j_given_i, get_psi, get_in_group_p_j_given_i
from src.transformation.ilr_clv_utils import inverse_ilr_transform, get_inode_relative_abundance
from src.transformation.l0_utils import hard_concrete_sample, hard_concrete_mean, l0_norm

EPS = 1e-6


class ilr_clv_transformation(transformation):
    def __init__(self, theta, Dv,
                 exist_in_group_dynamics=False, use_L0=True, inference_schedule=None,
                 reg_coef=1.0, b_dropout_rate=0.999,
                 training=False, annealing_frac=1.0):
        self.theta = theta
        self.Dx, self.Dy = theta.shape
        self.Dv = Dv
        self.use_L0 = use_L0
        self.inference_schedule = inference_schedule   # "flat" / "top_down" / "bottom_up"
        self.reg_coef = reg_coef
        self.annealing_frac = annealing_frac

        if self.inference_schedule == "flat":
            inode_b_init = 0.5
        elif self.inference_schedule == "top_down":
            inode_b_init = 1e-3
        elif self.inference_schedule == "bottom_up":
            inode_b_init = 1 - 1e-3
        else:
            raise NotImplementedError

        # init variable
        self.A_in_var = tf.Variable(tf.zeros((self.Dx, self.Dx + self.Dy)))
        self.g_in_var = tf.Variable(tf.zeros((self.Dx,)))
        self.Wv_in_var = tf.Variable(tf.zeros((self.Dx, self.Dv)))

        self.A_between_var = tf.Variable(tf.zeros((self.Dx, self.Dx + self.Dy)))
        self.g_between_var = tf.Variable(tf.zeros((self.Dx,)))
        self.Wv_between_var = tf.Variable(tf.zeros((self.Dx, self.Dv)))

        # init tree
        self.root, self.reference = convert_theta_to_tree(theta, inode_b_init)
        self.psi = get_psi(theta)

        inode_depths = get_inode_depths(self.Dx, self.reference)
        inode_heights = get_inode_heights(self.Dx, self.root)
        max_depth = np.max(inode_depths) + 1.0
        max_height = np.max(inode_heights) + 1.0

        if self.inference_schedule == "flat":
            self.training_mask = np.ones_like(inode_depths, dtype=bool)
        elif self.inference_schedule == "top_down":
            self.training_mask = (inode_depths * 0.75 / max_depth) < annealing_frac
        else:  # "bottom_up"
            self.training_mask = (inode_heights * 0.75 / max_height) < annealing_frac

        # L0 regularization for break score
        if use_L0:
            log_alpha_init = tf.log(b_dropout_rate / (1 - b_dropout_rate))
            self.log_alpha = tf.Variable(log_alpha_init * tf.ones(self.Dx, dtype=tf.float32))
            self.log_alpha = tf.where(self.training_mask, self.log_alpha, tf.stop_gradient(self.log_alpha))
            b_noises = tf.cond(training,
                               lambda: hard_concrete_sample(self.log_alpha),
                               lambda: hard_concrete_mean(self.log_alpha))
        else:
            b_noises = tf.ones(self.Dx, dtype=tf.float32)

        # training mask for b
        b = []
        b_before_gated = []
        for inode, b_noise, mask in zip(self.reference[:self.Dx], tf.unstack(b_noises), tf.unstack(self.training_mask)):
            b_before_gated.append(inode.b)
            inode.b = tf.where(mask, inode.b, tf.stop_gradient(inode.b))
            inode.b = inode.b * b_noise
            b.append(inode.b)
        self.b_before_gated = tf.stack(b_before_gated)
        self.b = tf.stack(b)

        # training mask for A_between
        self.A_between_var = self.get_A_between_var_masked()
        self.g_between_var = tf.where(self.training_mask, self.g_between_var, tf.stop_gradient(self.g_between_var))
        self.Wv_between_var = tf.where(self.training_mask, self.Wv_between_var, tf.stop_gradient(self.Wv_between_var))

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

        # get num_leaves for each node
        num_leaves = np.array([np.sum(theta_i != 0) for theta_i in self.theta])
        num_leaves = np.concatenate([num_leaves, np.ones(self.Dy)])

        L2 = tf.reduce_sum(self.A_in_var ** 2) / num_leaves + \
             tf.reduce_sum(self.g_in_var ** 2) + tf.reduce_sum(self.Wv_in_var ** 2) + \
             tf.reduce_sum(self.A_between_var ** 2) / num_leaves + \
             tf.reduce_sum(self.g_between_var ** 2) + tf.reduce_sum(self.Wv_between_var ** 2)
        L1 = tf.reduce_sum(tf.abs(self.A_in_var)) / num_leaves + \
             tf.reduce_sum(tf.abs(self.g_in_var)) + tf.reduce_sum(tf.abs(self.Wv_in_var)) + \
             tf.reduce_sum(tf.abs(self.A_between_var)) / num_leaves + \
             tf.reduce_sum(tf.abs(self.g_between_var)) + tf.reduce_sum(tf.abs(self.Wv_between_var))

        if self.inference_schedule == "flat":
            b_regularization = tf.reduce_sum(tf.log(1 - tf.abs(self.b_before_gated - 0.5) ** 3 + EPS))
        elif self.inference_schedule == "top_down":
            b_regularization = 0.0
        else:  # "bottom_up"
            b_regularization = tf.reduce_sum(tf.abs(self.b_before_gated))

        return ((L0 + L2) * self.reg_coef + b_regularization) * self.annealing_frac

    def transform(self, Input):
        """
        :param Input: (n_particles, batch_size, Dx + Dv)
        :param Dx: dimension of hidden space
        :return: output: (n_particles, batch_size, Dx)
        """
        # x_t + g_t + v_t * Wv + p_t * A

        A_in, g_in, Wv_in = tf.transpose(self.A_in, [1, 0]), self.g_in, tf.transpose(self.Wv_in, [1, 0])
        A_between, g_between, Wv_between = \
            tf.transpose(self.A_between, [1, 0]), self.g_between, tf.transpose(self.Wv_between, [1, 0])
        Dx = self.Dx

        x = Input[..., :Dx]  # (n_particles, batch_size, Dx)
        v = Input[0, 0:1, Dx:]  # (1, Dv)
        v_size = v.shape[-1]

        p_t = inverse_ilr_transform(x, self.psi)
        if self.inference_schedule == "bottom_up":
            # self-normalize inside each tree
            p_t = self.get_normalized_p_t(p_t)

        r_t = get_inode_relative_abundance(self.root, p_t, Dx)
        p_t = tf.concat([r_t, p_t], axis=-1)

        # (..., Dx + Dy, 1) * (Dx + Dy, Dx)
        pA = tf.reduce_sum(p_t[..., None] * (A_in + A_between), axis=-2)  # (..., Dx)

        delta = g_in + g_between + pA
        if v_size > 0:
            # Wv shape (Dv, Dx)
            Wvv = tf.reduce_sum(v[..., None] * (Wv_in + Wv_between), axis=-2)  # (n_particles, batch_size, Dx)
            delta += Wvv

        output = x + delta

        return output
    
    def get_A_between_var_masked(self):
        if self.inference_schedule == "flat":
            return self.A_between_var
        elif self.inference_schedule == "top_down":
            A_between_var_list = tf.unstack(self.A_between_var, axis=-1)
            A_between_var_masked = [0 for _ in range(self.Dx + self.Dy)]
            for inode, mask in zip(self.reference[:self.Dx], tf.unstack(self.training_mask)):
                idxes = [inode.left.node_idx, inode.right.node_idx]
                if inode == self.root:
                    idxes.append(inode.node_idx)
                for idx in idxes:
                    node_A_var = A_between_var_list[idx]
                    A_between_var_masked[idx] = tf.where(mask, node_A_var, tf.stop_gradient(node_A_var))
            return tf.stack(A_between_var_masked, axis=-1)
        else:  # "bottom_up"
            A_between_var_list = tf.unstack(self.A_between_var, axis=0)
            A_between_var_list = [tf.unstack(ele) for ele in A_between_var_list]
            A_between_var_masked = [[None for _ in range(self.Dx + self.Dy)] for _ in range(self.Dx)]
            for inode, mask in zip(self.reference[:self.Dx], tf.unstack(self.training_mask)):
                left_inode_idxes, left_taxon_idxes = get_inode_and_taxon_idxes(inode.left)
                right_inode_idxes, right_taxon_idxes = get_inode_and_taxon_idxes(inode.right)
                for i in left_inode_idxes:
                    for j in right_inode_idxes + right_taxon_idxes:
                        node_A_var = A_between_var_list[i][j]
                        A_between_var_masked[i][j] = tf.where(mask, node_A_var, tf.stop_gradient(node_A_var))
                for i in right_inode_idxes:
                    for j in left_inode_idxes + left_taxon_idxes:
                        node_A_var = A_between_var_list[i][j]
                        A_between_var_masked[i][j] = tf.where(mask, node_A_var, tf.stop_gradient(node_A_var))
                i = inode.inode_idx
                for j in left_taxon_idxes + right_taxon_idxes + [i]:
                    node_A_var = A_between_var_list[i][j]
                    A_between_var_masked[i][j] = tf.where(mask, node_A_var, tf.stop_gradient(node_A_var))
                for j in left_inode_idxes + right_inode_idxes:
                    node_A_var = A_between_var_list[i][j]
                    A_between_var_masked[i][j] = tf.where(mask, node_A_var, tf.stop_gradient(node_A_var))
                    node_A_var = A_between_var_list[j][i]
                    A_between_var_masked[j][i] = tf.where(mask, node_A_var, tf.stop_gradient(node_A_var))

            A_between_var_masked = [tf.stack(ele) for ele in A_between_var_masked]
            return tf.stack(A_between_var_masked)

    def get_bottom_up_subtrees(self):
        training_mask = tf.unstack(self.training_mask)

        is_root_of_subtree = []
        leaves_of_subtree = []

        for node in self.reference:
            if node == self.root:
                is_root_of_subtree.append(training_mask[node.inode_idx])
                leaves_of_subtree.append(list(range(self.Dy)))
                continue

            if node.is_taxon():
                mask = True
            else:
                mask = training_mask[node.node_idx]

            parent_mask = training_mask[node.parent.inode_idx]
            inode_is_root_of_subtree = tf.logical_and(mask, tf.logical_not(parent_mask))
            is_root_of_subtree.append(inode_is_root_of_subtree)

            _, child_taxon_idxes = get_inode_and_taxon_idxes(node)
            child_taxon_idxes = np.array(child_taxon_idxes) - self.Dx
            leaves_of_subtree.append(child_taxon_idxes)

        return is_root_of_subtree, leaves_of_subtree

    def get_normalized_p_t(self, p_t):

        p_t_list = tf.unstack(p_t, axis=-1)
        p_t_sum = tf.unstack(tf.ones_like(p_t), axis=-1)

        for is_root_of_subtree, leaves_of_subtree in zip(*self.get_bottom_up_subtrees()):
            p_t_sum_ = tf.reduce_sum([p_t_list[leaf_idx] for leaf_idx in leaves_of_subtree], axis=0)
            for leaf_idx in leaves_of_subtree:
                p_t_sum[leaf_idx] = tf.cond(is_root_of_subtree, lambda: p_t_sum_, lambda: p_t_sum[leaf_idx])

        p_t_sum = tf.stack(p_t_sum, axis=-1)
        return p_t / p_t_sum
