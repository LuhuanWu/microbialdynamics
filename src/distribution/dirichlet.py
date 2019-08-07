import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from distribution.base import distribution


class dirichlet(distribution):
    # dirichlet distribution

    def __init__(self, transformation):
        self.transformation = transformation

    def sample(self, Input):
        assert isinstance(Input, np.ndarray), "Input for dirichlet must be np.ndarray, {} is given".format(type(Input))

        def safe_softplus(x, limit=30):
            x[x < limit] = np.log(1.0 + np.exp(x[x < limit]))
            return x

        alphas = safe_softplus(self.transformation.transform(Input))
        return np.random.dirichlet(alphas)


class tf_dirichlet(distribution):
    # dirichlet distribution, can only be used as emission distribution

    def __init__(self, transformation, name='tf_dirichlet'):
        self.transformation = transformation
        self.name = name

    def get_dirichlet(self, Input):
        with tf.variable_scope(self.name):
            alphas, _ = self.transformation.transform(Input)
            alphas = tf.nn.softplus(alphas) + 1e-6
            dirichlet = tfd.Dirichlet(alphas,
                                      validate_args=True,
                                      allow_nan_stats=False)
            return dirichlet

    def log_prob(self, Input, output, name=None):
        dirichlet = self.get_dirichlet(Input)
        with tf.variable_scope(name or self.name):
            return dirichlet.log_prob(output)

    def mean(self, Input, name=None):
        dirichlet = self.get_dirichlet(Input)
        with tf.variable_scope(name or self.name):
            return dirichlet.mean()
