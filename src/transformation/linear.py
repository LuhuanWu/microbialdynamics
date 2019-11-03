import numpy as np
import tensorflow as tf

from src.transformation.base import transformation
from src.transformation.clv import batch_matmul

class linear_transformation(transformation):
    def transform(self, Input, **kwargs):
        '''
        Integrates the Lorenz ODEs
        '''
        A = self.params
        return np.dot(A, Input)


class tf_linear_transformation(transformation):
    def __init__(self, Dx, Dev):
        self.Dx = Dx
        self.Dev = Dev

        self.A = tf.Variable(tf.eye(self.Dx+self.Dev, self.Dx))
        self.b = tf.Variable(tf.zeros((self.Dx, )))

    def transform(self, Input, **kwargs):
        # Input shape: (n_particles, batch_size, Dx + Dev)
        output = batch_matmul(Input, self.A) + self.b

        return output