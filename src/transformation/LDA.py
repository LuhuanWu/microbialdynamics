import tensorflow as tf

from src.transformation.base import transformation, xavier_init
from src.transformation.clv import batch_matmul
from tensorflow.keras.layers import BatchNormalization


class LDA_transformation(transformation):
    def __init__(self, Dx, Dy, is_x_alr=True, training=False, beta_constant=True, beta_init_method='xavier'):
        self.Dx = Dx
        self.Dy = Dy
        self.is_x_lar = is_x_alr
        self.training = training
        self.beta_constant = beta_constant

        Din = Dx + (1 if is_x_alr else 0)

        sigma_con = tf.get_variable("sigma_con",
                                    shape=(Din, Dy),
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0),
                                    trainable=True)
        sigma_con = tf.nn.softplus(sigma_con)
        sigma_con = tf.where(tf.is_nan(sigma_con), tf.zeros_like(sigma_con), sigma_con)
        sigma_con = tf.maximum(sigma_con, 1e-6)

        if self.beta_constant:
            if beta_init_method == 'uniform':
                self.beta_log = tf.Variable(tf.ones((Din, Dy), dtype=tf.float32))
            elif beta_init_method == 'xavier':
                self.beta_log = tf.Variable(xavier_init(Din, Dy))
            else:
                raise ValueError("Unsupported beta_init_method: {}. "
                                 "Please choose from 'uniform', 'xavier'.".format(beta_init_method))

            self.beta_log_approx = self.beta_log + sigma_con * tf.random.normal((Din, Dy))

            self.batch_norm = BatchNormalization()
            #self.beta = tf.nn.softmax(self.batch_norm(self.beta_log_approx, training=training))
            # self.beta_mean = tf.nn.softmax(self.batch_norm(self.beta_log, training=False))
            self.beta = tf.nn.softmax(self.beta_log_approx)
            self.beta_mean = tf.nn.softmax(self.beta_log)

    def transform(self, x):
        if not self.beta_constant:
            assert isinstance(x, list), type(x)
            x, beta_log = x
            # x: (..., Dx), beta_log: (..., Dx+1, Dy-1). batch shape should match

        if self.is_x_lar:
            zeros = tf.zeros_like(x[..., 0:1])
            x = tf.concat([x, zeros], axis=-1)
        x = tf.nn.softmax(x, axis=-1)  #(..., Dx+1)

        # print(x.shape.as_list())
        if self.beta_constant:
            #output = batch_matmul(x, self.beta)

            # (..., Dx+1, 1) * (Dx+1, Dy)
            output = tf.reduce_sum(x[...,None]*self.beta, axis=-2) # (..., Dy)
        else:
            assert beta_log is not None
            zeros = tf.zeros_like(beta_log[..., 0:1])  # (..., Dx+1, 1)
            beta_log = tf.concat([beta_log, zeros], axis=-1)  # (..., Dx+1, Dy)
            beta = tf.nn.softmax(beta_log, axis=-1)

            # (..., Dx + 1,1) * (..., Dx+1, Dy)
            output = tf.reduce_sum(x[...,None]*beta, axis=-2)  # (..., Dy)
        output = tf.log(output)
        # assert output.shape[:-1] == x.shape[:-1]  # batch shape should match
        return output