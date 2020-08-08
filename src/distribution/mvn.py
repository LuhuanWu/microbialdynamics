import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from src.distribution.base import distribution


# np ver, just used in sampler, so no need to implement log_prob
class mvn(distribution):
    # multivariate normal distribution

    def __init__(self, transformation, name="mvn", sigma=1):
        super(mvn, self).__init__(transformation, name)
        self.sigmaChol = np.linalg.cholesky(sigma)

    def sample(self, Input):
        mu = self.transformation.transform(Input)
        return mu + np.dot(self.sigmaChol, np.random.randn(len(mu)))


# tf ver, used in calculate log_ZSMC
class tf_mvn(distribution):
    # multivariate normal distribution

    def __init__(self, transformation, name='tf_mvn', sigma_init=5, sigma_min=1, rank=1, train_sigma=True):
        super(tf_mvn, self).__init__(transformation, name)
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.rank = rank  # rank of the random variable
        self.train_sigma = train_sigma


    def get_mvn(self, Input):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            mu = self.transformation.transform(Input)
            sigma_con = self.get_sigma(mu)
            mvn = tfd.MultivariateNormalDiag(mu, sigma_con,
                                             validate_args=True,
                                             allow_nan_stats=False)
            return mvn

    def get_sigma(self, mu):
        shape = [mu.shape.as_list()[-self.rank+i] for i in range(self.rank)]
        sigma_con = tf.get_variable("sigma_con",
                                    shape=shape,
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(self.sigma_init),
                                    trainable=self.train_sigma)
        sigma_con = tf.nn.softplus(sigma_con)
        sigma_con = tf.where(tf.is_nan(sigma_con), tf.zeros_like(sigma_con), sigma_con)
        sigma_con = tf.maximum(sigma_con, self.sigma_min)
        self.sigma_con = sigma_con
        return sigma_con

    def sample_and_log_prob(self, Input, sample_shape=(), name=None, **kwargs):
        mvn = self.get_mvn(Input)
        with tf.variable_scope(name or self.name):
            sample = mvn.sample(sample_shape)
            log_prob = mvn.log_prob(sample)
            return sample, log_prob

    def sample(self, Input, sample_shape=(), name=None, **kwargs):
        mvn = self.get_mvn(Input)
        with tf.variable_scope(name or self.name):
            sample = mvn.sample(sample_shape)
            return sample

    def log_prob(self, Input, output, name=None, **kwargs):
        mvn = self.get_mvn(Input)
        with tf.variable_scope(name or self.name):
            return mvn.log_prob(output)

    def mean(self, Input, name=None, **kwargs):
        mvn = self.get_mvn(Input)
        with tf.variable_scope(name or self.name):
            return mvn.mean()

    def sigma(self, Input, name=None, **kwargs):
        mvn = self.get_mvn(Input)
        with tf.variable_scope(name or self.name):
            return mvn.stddev()
