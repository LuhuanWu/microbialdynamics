import tensorflow as tf

from src.transformation.base import transformation, xavier_init
from src.transformation.clv import batch_matmul
from tensorflow.keras.layers import BatchNormalization


class LDA_transformation(transformation):
    def __init__(self, Dx, Dy, is_x_alr=True, training=False, beta_init_method='uniform'):
        self.Dx = Dx
        self.Dy = Dy
        self.is_x_lar = is_x_alr
        self.training = training

        Din = Dx + (1 if is_x_alr else 0)
        if beta_init_method == 'uniform':
            self.beta_alr = tf.Variable(tf.ones((Din, Dy), dtype=tf.float32))
        elif beta_init_method == 'xavier':
            self.beta_alr = tf.Variable(xavier_init(Din, Dy))
        else:
            raise ValueError("Unsupported beta_init_method: {}. "
                             "Please choose from 'uniform', 'xavier'.".format(beta_init_method))
        self.batch_norm = BatchNormalization()
        self.beta = tf.nn.softmax(self.batch_norm(self.beta_alr, training=training))

    def transform(self, x):
        if self.is_x_lar:
            zeros = tf.zeros_like(x[..., 0:1])
            x = tf.concat([x, zeros], axis=-1)
        x = tf.nn.softmax(x, axis=-1)

        # print(x.shape.as_list())
        output = batch_matmul(x, self.beta)
        output = tf.log(output)
        return output