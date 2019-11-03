import tensorflow as tf

from src.transformation.base import transformation, xavier_init
from src.transformation.clv import batch_matmul


class LDA_transformation(transformation):
    def __init__(self, Dx, Dy, is_f_clv=False):
        self.Dx = Dx
        self.Dy = Dy
        self.is_f_clv = is_f_clv

        Din = Dx + (1 if is_f_clv else 0)
        self.beta = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.Variable(xavier_init(Din, Dy))))

    def transform(self, x, **kwargs):
        zeros = tf.zeros_like(x[..., 0:1])
        x = tf.concat([x, zeros], axis=-1)
        x = tf.nn.softmax(x, axis=-1)

        # print(x.shape.as_list())
        output = batch_matmul(x, self.beta)
        output = tf.log(output)
        return output