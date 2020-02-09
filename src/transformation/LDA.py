import tensorflow as tf

from src.transformation.base import transformation, xavier_init
from src.transformation.clv import batch_matmul
from tensorflow.keras.layers import BatchNormalization


class LDA_transformation(transformation):
    def __init__(self, Dx, Dy, is_x_alr=True, training=False, beta_init_method='xavier'):
        self.Dx = Dx
        self.Dy = Dy
        self.is_x_lar = is_x_alr
        self.training = training

        Din = Dx + (1 if is_x_alr else 0)
        if beta_init_method == 'uniform':
            self.beta_log = tf.Variable(tf.ones((Din, Dy), dtype=tf.float32))
        elif beta_init_method == 'xavier':
            self.beta_log = tf.Variable(xavier_init(Din, Dy))
        else:
            raise ValueError("Unsupported beta_init_method: {}. "
                             "Please choose from 'uniform', 'xavier'.".format(beta_init_method))

        sigma_con = tf.get_variable("sigma_con",
                                    shape=(Din, Dy),
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0),
                                    trainable=True)
        sigma_con = tf.nn.softplus(sigma_con)
        sigma_con = tf.where(tf.is_nan(sigma_con), tf.zeros_like(sigma_con), sigma_con)
        sigma_con = tf.maximum(sigma_con, 1e-6)

        self.beta_log_approx = self.beta_log + sigma_con * tf.random.normal((Din, Dy))

        self.batch_norm = BatchNormalization()
        # self.beta = tf.nn.softmax(self.batch_norm(self.beta_log_approx, training=training))
        # self.beta_mean = tf.nn.softmax(self.batch_norm(self.beta_log, training=False))
        self.beta = tf.nn.softmax(self.beta_log_approx)
        self.beta_mean = tf.nn.softmax(self.beta_log)

    def transform(self, x):
        if self.is_x_lar:
            zeros = tf.zeros_like(x[..., 0:1])
            x = tf.concat([x, zeros], axis=-1)
        x = tf.nn.softmax(x, axis=-1)

        # print(x.shape.as_list())
        output = batch_matmul(x, self.beta)
        output = tf.log(output)
        return output