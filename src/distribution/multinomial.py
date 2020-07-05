import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from src.distribution.base import distribution


class tf_multinomial(distribution):
    # multivariate multinomial distribution, can only be used as emission distribution

    def __init__(self, transformation, name='tf_multinomial'):
        super(tf_multinomial, self).__init__(transformation, name)

    def get_multinomial(self, Input, depth):
        """
        :param Input: (T, Dx)
        :param external_inputs:  total counts, (T, )
        :return:
        """
        with tf.variable_scope(self.name):
            logits = self.transformation.transform(Input)
            multinomial = tfd.Multinomial(total_count=depth,
                                          logits=logits,
                                          validate_args=True,
                                          allow_nan_stats=False)
            return multinomial

    def log_prob(self, Input, output, name=None):
        depth = tf.reduce_sum(output, axis=-1)
        multinomial = self.get_multinomial(Input, depth)
        with tf.variable_scope(name or self.name):
            return multinomial.log_prob(output)

    def mean(self, Input, obs, name=None):
        depth = tf.reduce_sum(obs, axis=-1)
        multinomial = self.get_multinomial(Input, depth)
        with tf.variable_scope(name or self.name):
            return multinomial.mean()
