import tensorflow as tf

from src.transformation.base import transformation, xavier_init
from src.transformation.clv import batch_matmul
from tensorflow.keras.layers import BatchNormalization


class LDA_transformation(transformation):
    def __init__(self, Dx, Dy, is_x_lar=True, training=False):
        self.Dx = Dx
        self.Dy = Dy
        self.is_x_lar = is_x_lar
        self.training = training

        Din = Dx + (1 if is_x_lar else 0)
        self.beta_lar = tf.Variable(xavier_init(Din, Dy))
        self.batch_norm = BatchNormalization()
        self.beta = tf.nn.softmax(self.batch_norm(self.beta_lar, training=training))

    def transform(self, x):
        if self.is_x_lar:
            zeros = tf.zeros_like(x[..., 0:1])
            x = tf.concat([x, zeros], axis=-1)
        x = tf.nn.softmax(x, axis=-1)

        # print(x.shape.as_list())
        output = batch_matmul(x, self.beta)
        output = tf.log(output)
        return output