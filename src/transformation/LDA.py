import tensorflow as tf

from src.transformation.base import transformation, xavier_init


class LDA_transformation(transformation):
    def __init__(self, Dx, Dy, beta_init_method='xavier'):
        self.Dx = Dx
        self.Dy = Dy

        if beta_init_method == 'uniform':
            self.beta_log = tf.Variable(tf.ones((Dx, Dy), dtype=tf.float32))
        elif beta_init_method == 'xavier':
            self.beta_log = tf.Variable(xavier_init(Dx, Dy))
        else:
            raise ValueError("Unsupported beta_init_method: {}. "
                             "Please choose from 'uniform', 'xavier'.".format(beta_init_method))

        log_sigma_con = tf.get_variable("log_sigma_con",
                                        shape=(Dx, Dy),
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(1.0),
                                        trainable=True)
        sigma_con = tf.exp(tf.clip_by_value(log_sigma_con, -8, 8))

        self.beta_log_approx = self.beta_log + sigma_con * tf.random.normal((Dx, Dy))
        self.beta = tf.nn.softmax(self.beta_log_approx)
        self.beta_mean = tf.nn.softmax(self.beta_log)

    def transform(self, x):
        x = tf.nn.softmax(x, axis=-1)  # (..., Dx+1)

        # (..., Dx+1, 1) * (Dx+1, Dy)
        output = tf.reduce_sum(x[..., None] * self.beta, axis=-2)  # (..., Dy)

        return output