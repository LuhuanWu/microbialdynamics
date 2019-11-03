import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from src.distribution.base import distribution


class poisson(distribution):
    # multivariate poisson distribution

    def __init__(self, transformation, name="poisson"):
        super(poisson, self).__init__(transformation, name)

    def sample(self, Input, **kwargs):
        assert isinstance(Input, np.ndarray), "Input for poisson must be np.ndarray, {} is given".format(type(Input))

        def safe_softplus(x, limit=30):
            x[x < limit] = np.log(1.0 + np.exp(x[x < limit]))
            return x

        lambdas = safe_softplus(self.transformation.transform(Input, **kwargs))
        return np.random.poisson(lambdas)


class tf_poisson(distribution):
    # multivariate poisson distribution, can only be used as emission distribution

    def __init__(self, transformation, name='tf_poisson'):
        super(tf_poisson, self).__init__(transformation, name)


    def get_poisson(self, Input, extra_inputs=None):
        """
        :param Input: (T, Dx)
        :param external_inputs:  total counts, (T, )
        :return:
        """
        with tf.variable_scope(self.name):
            lambdas = self.transformation.transform(Input)
            lambdas = tf.nn.softplus(lambdas) + 1e-6  # (T, Dy)
            lambdas = lambdas / tf.reduce_sum(lambdas, axis=-1, keepdims=True)
            lambdas = lambdas * extra_inputs[..., None]  # (bs, T, Dy)
            poisson = tfd.Poisson(rate=lambdas,
                                  validate_args=True,
                                  allow_nan_stats=False)
            return poisson

    def log_prob(self, Input, output, extra_inputs=None, name=None):
        poisson = self.get_poisson(Input, extra_inputs)
        with tf.variable_scope(name or self.name):
            return tf.reduce_sum(poisson.log_prob(output), axis=-1)

    def mean(self, Input, extra_inputs, name=None):
        poisson = self.get_poisson(Input, extra_inputs)
        with tf.variable_scope(name or self.name):
            return poisson.mean()
