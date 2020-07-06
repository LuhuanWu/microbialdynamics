import numpy as np
from copy import deepcopy

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from src.distribution.base import distribution
from src.transformation.inv_ilr import inv_ilr_transformation


class tf_multinomial_compose(distribution):
    # multivariate multinomial distribution, can only be used as emission distribution

    def __init__(self, transformation, name='tf_multinomial'):
        super(tf_multinomial_compose, self).__init__(transformation, name)
        assert isinstance(transformation, inv_ilr_transformation)
        assert transformation.ilr_clv is not None
        self.ilr_clv = transformation.ilr_clv

    def log_prob(self, Input, output, name=None):
        output_list = tf.unstack(output, axis=-1)
        logits = self.transformation.transform(Input)
        logits_list = tf.unstack(logits, axis=-1)
        log_prob = tf.zeros_like(logits[..., 0])

        def log_prob_i(leaves_of_subtree_):
            if len(leaves_of_subtree_) == 1:
                return tf.zeros_like(logits[..., 0])
            with tf.variable_scope(name or self.name):
                output_ = tf.stack([output_list[leaf_idx] for leaf_idx in leaves_of_subtree_], axis=-1)
                logits_ = tf.stack([logits_list[leaf_idx] for leaf_idx in leaves_of_subtree_], axis=-1)
                depth = tf.reduce_sum(output_, axis=-1)
                multinomial = tfd.Multinomial(total_count=depth, logits=logits_,
                                              validate_args=True, allow_nan_stats=False)
                log_prob_i = multinomial.log_prob(output_)
            return log_prob_i

        for is_root_of_subtree, leaves_of_subtree in zip(*self.ilr_clv.get_bottom_up_subtrees()):
            log_prob_i_ = tf.cond(is_root_of_subtree,
                                  lambda: log_prob_i(leaves_of_subtree),
                                  lambda: tf.zeros_like(logits[..., 0]))
            log_prob += log_prob_i_

        return log_prob

    def mean(self, Input, obs, name=None):
        obs_list = tf.unstack(obs, axis=-1)
        logits = self.transformation.transform(Input)
        logits_list = tf.unstack(logits, axis=-1)
        Dy = obs.shape.as_list()[-1]

        mean = [0.0 for _ in range(Dy)]

        def get_mean_subtree(leaves_of_subtree_):
            if len(leaves_of_subtree_) == 1:
                leaf_idx_ = leaves_of_subtree_[0]
                return [obs_list[leaf_idx_]]

            with tf.variable_scope(name or self.name):
                logits_ = tf.stack([logits_list[leaf_idx_] for leaf_idx_ in leaves_of_subtree_], axis=-1)
                depth = tf.reduce_sum([obs_list[leaf_idx_] for leaf_idx_ in leaves_of_subtree_], axis=0)
                multinomial = tfd.Multinomial(total_count=depth, logits=logits_,
                                              validate_args=True, allow_nan_stats=False)
                mean_subtree = multinomial.mean()
            return tf.unstack(mean_subtree, axis=-1)

        for is_root_of_subtree, leaves_of_subtree in zip(*self.ilr_clv.get_bottom_up_subtrees()):
            mean_subtree = tf.cond(is_root_of_subtree,
                                   lambda: get_mean_subtree(leaves_of_subtree),
                                   lambda: [mean[leaf_idx] for leaf_idx in leaves_of_subtree])
            if len(leaves_of_subtree) == 1:
                # for some reason, tf.cond will automatically squeeze lambda return if it's a list of len == 1
                mean_subtree = [mean_subtree]
            for i, leaf_idx in enumerate(leaves_of_subtree):
                mean[leaf_idx] = mean_subtree[i]

        return tf.stack(mean, axis=-1)
