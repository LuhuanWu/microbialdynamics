import tensorflow as tf


class inv_alr_transformation(object):
    # base class for transformation
    def __init__(self, params=None):
        self.params = params

    def transform(self, x):
        zeros = tf.zeros_like(x[..., 0:1])
        x = tf.concat([x, zeros], axis=-1)
        y = tf.nn.softmax(x, axis=-1)
        y = tf.log(y)
        return y
