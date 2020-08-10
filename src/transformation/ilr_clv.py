import numpy as np
import tensorflow as tf

from src.transformation.base import transformation
from src.transformation.ilr_clv_utils import convert_theta_to_tree
from src.transformation.ilr_clv_utils import setup_heights, get_inode_and_taxon_idxes
from src.transformation.ilr_clv_utils import get_psi, inverse_ilr_transform, get_inode_relative_abundance
from src.transformation.l0_utils import hard_concrete_sample, hard_concrete_mean, l0_norm, GAMMA, ZETA, BETA

EPS = 1e-6


class ilr_clv_transformation(transformation):
    def __init__(self, theta, Dv, flat_inference=False,
                 reg_coef=1.0, params_reg_func="L2", overlap_reg_func="L1",
                 training=False, annealing_frac=1.0, in_training_delay=0.0, num_leaves_sum=True):
        self.theta = theta
        Dx, Dy = self.Dx, self.Dy = theta.shape
        self.Dv = Dv
        self.flat_inference = flat_inference
        self.training = training
        self.annealing_frac = annealing_frac

        assert params_reg_func in ["L1", "L2"]
        assert overlap_reg_func in ["L1", "KL", "L2", "None"]
        self.params_reg_func = params_reg_func
        self.overlap_reg_func = overlap_reg_func
        self.reg_coef = reg_coef

        # init tree
        self.root, self.reference = convert_theta_to_tree(self.theta)
        setup_heights(self.root)
        self.psi = get_psi(self.theta)

        # min depth (root) is 1
        self.depths = np.array([node.depth for node in self.reference])
        # min inode height (inode whose children are all taxon) is 1
        self.heights = np.array([node.height for node in self.reference])
        self.inode_heights = self.heights[:Dx]
        self.parent_heights = np.array([node.parent.height if node.parent is not None else node.height
                                        for node in self.reference])
        if num_leaves_sum:
            num_leaves = np.array([np.sum(theta_i == 1) + np.sum(theta_i == -1) for theta_i in self.theta])
        else:
            num_leaves = np.array([np.sum(theta_i == 1) * np.sum(theta_i == -1) for theta_i in self.theta])
        self.num_leaves = np.concatenate([num_leaves, np.ones(Dy)])

        # training mask
        if self.flat_inference:
            self.in_training_mask = np.ones(Dx, dtype=bool)
            self.between_training_mask = np.ones(Dx + Dy, dtype=bool)
            self.in_reg_annealing = np.ones(Dx) * self.annealing_frac
            self.between_reg_annealing = np.ones(Dx + Dy) * self.annealing_frac
        else:
            schedule_frac = 0.6  # how long it takes for top-down & bottom-up to fully spread to all nodes

            in_training_starts = ((self.inode_heights - 1) / self.inode_heights.max()) * schedule_frac
            in_training_starts += in_training_delay

            max_parent_height = self.parent_heights.max()
            between_training_starts = ((max_parent_height - self.parent_heights) / max_parent_height) * schedule_frac

            self.in_training_starts, self.between_training_starts = in_training_starts, between_training_starts

            self.in_training_mask = in_training_starts < annealing_frac
            self.between_training_mask = between_training_starts < annealing_frac
            self.in_reg_annealing = (self.annealing_frac - in_training_starts) / (1.0 - in_training_starts)
            self.in_reg_annealing = tf.maximum(0.0, self.in_reg_annealing)
            self.between_reg_annealing = (self.annealing_frac - between_training_starts) / \
                                         (1.0 - between_training_starts)
            self.between_reg_annealing = tf.maximum(0.0, self.between_reg_annealing)

        # init variable
        self.A_in_var = tf.Variable(tf.zeros((Dx, Dy)))
        self.g_in_var = tf.Variable(tf.zeros((Dx,)))
        self.Wv_in_var = tf.Variable(tf.zeros((Dx, self.Dv)))

        self.A_between_var = tf.Variable(tf.zeros((Dx, Dx + Dy)))
        self.g_between_var = tf.Variable(tf.zeros((Dx,)))
        self.Wv_between_var = tf.Variable(tf.zeros((Dx, self.Dv)))

        # init in-group / between-group assignment
        assignment_init = 0.7
        assignment_init = tf.log(assignment_init / (1 - assignment_init))
        self.in_assignment_var = tf.Variable(assignment_init * tf.ones(Dx, dtype=tf.float32))
        self.between_assignment_var = tf.Variable(assignment_init * tf.ones(Dx + Dy, dtype=tf.float32))

        self.in_assignment_var = tf.where(self.in_training_mask,
                                          self.in_assignment_var,
                                          -10.0 * tf.ones_like(self.in_assignment_var))
        self.between_assignment_var = tf.where(self.between_training_mask,
                                               self.between_assignment_var,
                                               -10.0 * tf.ones_like(self.between_assignment_var))

        self.in_assignment_before_gated = tf.sigmoid(self.in_assignment_var)
        self.between_assignment_before_gated = tf.sigmoid(self.between_assignment_var)

        # L0 regularization for between-group / in-group assignment
        L0_init = 0.7
        log_alpha_init = (L0_init - GAMMA) / (ZETA - GAMMA)
        log_alpha_init = tf.log(log_alpha_init / (1 - log_alpha_init)) * BETA
        self.in_log_alpha = tf.Variable(log_alpha_init * tf.ones(Dx, dtype=tf.float32))
        self.between_log_alpha = tf.Variable(log_alpha_init * tf.ones(Dx + Dy, dtype=tf.float32))

        self.in_log_alpha = tf.where(self.in_training_mask, self.in_log_alpha, -10.0 * tf.ones_like(self.in_log_alpha))
        self.in_L0_noise = tf.cond(self.training,
                                   lambda: hard_concrete_sample(self.in_log_alpha),
                                   lambda: hard_concrete_mean(self.in_log_alpha))

        self.between_log_alpha = tf.where(self.between_training_mask,
                                          self.between_log_alpha,
                                          -10.0 * tf.ones_like(self.between_log_alpha))
        self.between_L0_noise = tf.cond(self.training,
                                        lambda: hard_concrete_sample(self.between_log_alpha),
                                        lambda: hard_concrete_mean(self.between_log_alpha))

        # apply assignment
        self.in_assignment = self.in_assignment_before_gated * self.in_L0_noise
        self.between_assignment = self.between_assignment_before_gated * self.between_L0_noise

        self.A_in_var, self.A_in = self.get_A_in(self.A_in_var, self.in_training_mask, self.in_assignment)
        self.g_in_var = tf.where(self.in_training_mask, self.g_in_var, tf.stop_gradient(self.g_in_var))
        self.Wv_in_var = tf.where(self.in_training_mask, self.Wv_in_var, tf.stop_gradient(self.Wv_in_var))
        self.g_in = self.g_in_var * self.in_assignment
        self.Wv_in = self.Wv_in_var * self.in_assignment[:, None]

        self.A_between_var, self.A_between = \
            self.get_A_between(self.A_between_var, self.between_training_mask, self.between_assignment)
        self.g_between_var = \
            tf.where(self.between_training_mask[:Dx], self.g_between_var, tf.stop_gradient(self.g_between_var))
        self.Wv_between_var = \
            tf.where(self.between_training_mask[:Dx], self.Wv_between_var, tf.stop_gradient(self.Wv_between_var))
        self.g_between = self.g_between_var * self.between_assignment[:Dx]
        self.Wv_between = self.Wv_between_var * self.between_assignment[:Dx, None]

        self.params = {"in_assignment": self.in_assignment, "between_assignment": self.between_assignment,
                       "A_in": self.A_in, "g_in": self.g_in, "Wv_in": self.Wv_in,
                       "A_between": self.A_between, "g_between": self.g_between, "Wv_between": self.Wv_between}

    def regularization_loss(self):
        # regularize in-group params by height (exponentially), and between-group params by num_leaves (linearly)
        Dx = self.Dx
        num_leaves = self.num_leaves
        inode_num_leaves = num_leaves[:Dx]

        in_L0 = l0_norm(self.in_log_alpha)
        self.in_L0 = tf.reduce_sum(in_L0 * self.in_reg_annealing * inode_num_leaves)
        between_L0 = l0_norm(self.between_log_alpha)
        self.between_L0 = tf.reduce_sum(between_L0 * self.between_reg_annealing / num_leaves)
        L0 = self.in_L0 + self.between_L0

        in_assigment_reg = tf.reduce_sum(self.in_assignment_before_gated * self.in_reg_annealing * inode_num_leaves)
        between_assigment_reg = \
            tf.reduce_sum(self.between_assignment_before_gated * self.between_reg_annealing / num_leaves)
        assigment_reg = in_assigment_reg + between_assigment_reg

        if self.params_reg_func == "L1":
            params_reg = tf.reduce_sum(self.A_between_L1) + \
                         tf.reduce_sum(tf.abs(self.g_between_var) * (self.between_reg_annealing / num_leaves)[:Dx]) + \
                         tf.reduce_sum(tf.abs(self.Wv_between_var) *
                                       (self.between_reg_annealing / num_leaves)[:Dx, None]) + \
                         tf.reduce_sum(self.A_in_L2) + \
                         tf.reduce_sum(tf.abs(self.g_in_var) * self.in_reg_annealing * inode_num_leaves) + \
                         tf.reduce_sum(tf.abs(self.Wv_in_var) * (self.in_reg_annealing * inode_num_leaves)[:, None])
        else:  # L2
            params_reg = tf.reduce_sum(self.A_between_L2) + \
                         tf.reduce_sum(self.g_between_var ** 2 * (self.between_reg_annealing / num_leaves)[:Dx]) + \
                         tf.reduce_sum(self.Wv_between_var ** 2 *
                                       (self.between_reg_annealing / num_leaves)[:Dx, None]) + \
                         tf.reduce_sum(self.A_in_L1) + \
                         tf.reduce_sum(self.g_in_var ** 2 * self.in_reg_annealing * inode_num_leaves) + \
                         tf.reduce_sum(self.Wv_in_var ** 2 * (self.in_reg_annealing * inode_num_leaves)[:, None])

        overlap_reg = 0.0
        between_assigment = tf.unstack(self.between_assignment)
        for inode, a_in, num_leaves_i in zip(self.reference[:Dx], tf.unstack(self.in_assignment), inode_num_leaves):
            child_inode_idxes, child_taxon_idxes = get_inode_and_taxon_idxes(inode)
            a_in = tf.clip_by_value(a_in, EPS, 1 - EPS)
            for idx in child_inode_idxes + child_taxon_idxes:
                child_a_between = between_assigment[idx]
                child_a_between = tf.clip_by_value(child_a_between, EPS, 1 - EPS)
                num_leaves_j = num_leaves[idx]
                if self.overlap_reg_func == "L1":
                    overlap_reg_ij = a_in * child_a_between
                elif self.overlap_reg_func == "L2":
                    overlap_reg_ij = (a_in * child_a_between) ** 2
                elif self.overlap_reg_func == "KL":
                    overlap_reg_ij = -a_in * (tf.log(a_in) - tf.log(child_a_between))
                else:  # None
                    overlap_reg_ij = tf.constant(0.0, dtype=tf.float32)
                overlap_reg += overlap_reg_ij * num_leaves_i / num_leaves_j

        with tf.variable_scope('reg_loss'):
            tf.summary.scalar('L0', L0 * self.reg_coef)
            tf.summary.scalar('assigment_reg', assigment_reg * self.reg_coef)
            tf.summary.scalar('params_reg', params_reg * self.reg_coef)
            tf.summary.scalar('overlap_reg', overlap_reg * self.reg_coef)

        return (L0 + assigment_reg + params_reg + overlap_reg) * self.reg_coef

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
        r_t = get_inode_relative_abundance(self.root, p_t, Dx)
        r_p_t = tf.concat([r_t, p_t], axis=-1)

        # (..., Dx)
        pA = tf.reduce_sum(p_t[..., None] * A_in, axis=-2) + tf.reduce_sum(r_p_t[..., None] * A_between, axis=-2)

        delta = g_in + g_between + pA
        if v_size > 0:
            # Wv shape (Dv, Dx)
            Wvv = tf.reduce_sum(v[..., None] * (Wv_in + Wv_between), axis=-2)  # (..., Dx)
            delta += Wvv

        output = x + delta

        return output

    def get_A_in(self, A_in_var, in_training_mask, in_assignment):
        inodes = self.reference[:self.Dx]
        in_training_mask = tf.unstack(in_training_mask)
        in_reg_annealing = tf.unstack(self.in_reg_annealing)
        in_assignment = tf.unstack(in_assignment)
        A_in_var_list = tf.unstack(A_in_var, axis=0)
        A_in_var_list = [tf.unstack(ele) for ele in A_in_var_list]
        A_in_var_masked = [[None for _ in range(self.Dy)] for _ in range(self.Dx)]
        A_in_L1 = [[None for _ in range(self.Dy)] for _ in range(self.Dx)]
        A_in_L2 = [[None for _ in range(self.Dy)] for _ in range(self.Dx)]
        A_in = [[None for _ in range(self.Dy)] for _ in range(self.Dx)]

        for inode, mask, reg_annealing, num_leaves, assignment in \
                zip(inodes, in_training_mask, in_reg_annealing, self.num_leaves[:self.Dx], in_assignment):
            left_inode_idxes, left_taxon_idxes = get_inode_and_taxon_idxes(inode.left)
            right_inode_idxes, right_taxon_idxes = get_inode_and_taxon_idxes(inode.right)
            idxes_to_update = [[left_inode_idxes, right_taxon_idxes],
                               [right_inode_idxes, left_taxon_idxes],
                               [[inode.inode_idx], left_taxon_idxes + right_taxon_idxes], ]
            for i_idxes, j_idxes in idxes_to_update:
                for i in i_idxes:
                    for j in j_idxes:
                        j -= self.Dx
                        node_A_var = A_in_var_list[i][j]
                        node_A_var = tf.where(mask, node_A_var, tf.stop_gradient(node_A_var))
                        A_in_var_masked[i][j] = node_A_var
                        A_in_L1[i][j] = tf.abs(node_A_var) * reg_annealing * num_leaves
                        A_in_L2[i][j] = node_A_var ** 2 * reg_annealing * num_leaves
                        A_in[i][j] = node_A_var * assignment

        A_in_var_masked = tf.stack([tf.stack(ele) for ele in A_in_var_masked])
        self.A_in_L1 = tf.stack([tf.stack(ele) for ele in A_in_L1])
        self.A_in_L2 = tf.stack([tf.stack(ele) for ele in A_in_L2])
        A_in = tf.stack([tf.stack(ele) for ele in A_in])
        return A_in_var_masked, A_in

    def get_A_between(self, A_between_var, between_training_mask, between_assignment):
        Dx, Dy = self.Dx, self.Dy

        between_training_mask = tf.unstack(between_training_mask)
        between_assignment = tf.unstack(between_assignment)

        between_reg_annealing = tf.unstack(self.between_reg_annealing)
        num_leaves = self.num_leaves

        A_between_var_list = tf.unstack(A_between_var, axis=0)
        A_between_var_list = [tf.unstack(ele) for ele in A_between_var_list]
        A_between_var_masked = [[None for _ in range(Dx + Dy)] for _ in range(Dx)]
        A_between_L1 = [[None for _ in range(Dx + Dy)] for _ in range(Dx)]
        A_between_L2 = [[None for _ in range(Dx + Dy)] for _ in range(Dx)]
        A_between = [[None for _ in range(Dx + Dy)] for _ in range(Dx)]

        for i, (mask_i, reg_annealing_i, num_leaves_i, assignment_i) in \
                enumerate(zip(between_training_mask[:Dx], between_reg_annealing[:Dx],
                              num_leaves[:Dx], between_assignment[:Dx])):
            for j, (mask_j, reg_annealing_j, num_leaves_j, assignment_j) in \
                    enumerate(zip(between_training_mask, between_reg_annealing, num_leaves, between_assignment)):
                mask = tf.logical_and(mask_i, mask_j)
                reg_annealing = tf.minimum(reg_annealing_i, reg_annealing_j)
                num_leaves_ = min(num_leaves_i, num_leaves_j)
                node_A_var = A_between_var_list[i][j]
                node_A_var = tf.where(mask, node_A_var, tf.stop_gradient(node_A_var))
                A_between_var_masked[i][j] = node_A_var
                A_between_L1[i][j] = tf.abs(node_A_var) * reg_annealing / num_leaves_
                A_between_L2[i][j] = node_A_var ** 2 * reg_annealing / num_leaves_
                A_between[i][j] = node_A_var * assignment_i * assignment_j

        A_between_var_masked = tf.stack([tf.stack(ele) for ele in A_between_var_masked])
        self.A_between_L1 = tf.stack([tf.stack(ele) for ele in A_between_L1])
        self.A_between_L2 = tf.stack([tf.stack(ele) for ele in A_between_L2])
        A_between = tf.stack([tf.stack(ele) for ele in A_between])
        return A_between_var_masked, A_between

