import numpy as np
import tensorflow as tf

from src.transformation.base import transformation

class linear_transformation(transformation):
    def transform(self, Input, **kwargs):
        '''
        Integrates the Lorenz ODEs
        '''
        A = self.params
        return np.dot(A, Input)


class tf_linear_transformation(transformation):
    def __init__(self, Dx, Dv):
        self.Dx = Dx
        self.Dv = Dv

        self.A = tf.Variable(tf.eye(self.Dx+self.Dv, self.Dx))
        self.b = tf.Variable(tf.zeros((self.Dx, )))

    def transform(self, Input):
        # Input shape: (n_particles, batch_size, Dx + Dv)
        output = tf.reduce_sum(Input[..., None] * self.A, axis=-2) + self.b

        return output