import numpy as np
import tensorflow as tf

from src.transformation.base import transformation


class inver_lar_transformation(transformation):
    def transform(self, Input, **kwargs):
        '''
        # Input shape: (n_particles, batch_size, Dy - 1)
        # return log percentage: (n_particles, batch_size, Dy)
        '''

        x_last = tf.zeros_like(Input)[..., 0:1]  # (n_particles, batch_size, 1)
        x = tf.concat((Input, x_last), axis=-1)  # (n_particles, batch_size, Dy)

        log_p = x - tf.reduce_logsumexp(x, axis=-1, keepdims=True)  # (n_particles, batch_size, Dy)
        return log_p, 0


