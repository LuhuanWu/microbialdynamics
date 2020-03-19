import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.initializers import Orthogonal

from src.transformation.base import transformation


class MLP_transformation(transformation):
    def __init__(self, Dhs, Dout,
                 use_residual=False,
                 training=False,
                 initialize_around_zero=False,
                 name="MLP_transformation"):
        self.Dhs = Dhs
        self.Dout = Dout

        self.use_residual = use_residual
        self.training = training
        self.initialize_around_zero = initialize_around_zero

        self.name = name
        self.init_FFN()

    def init_FFN(self):
        with tf.variable_scope(self.name):
            self.hidden_layers = []
            for i, Dh in enumerate(self.Dhs):
                self.hidden_layers.append(
                    Dense(Dh,
                          activation="relu",
                          kernel_initializer="he_normal",
                          name="hidden_{}".format(i))
                )

            mu_initializer = Orthogonal(0.01) if self.initialize_around_zero else "he_normal"
            self.mu_layer = Dense(self.Dout,
                                  activation="linear",
                                  kernel_initializer=mu_initializer,
                                  name="mu_layer")

            if self.use_residual:
                self.batch_norm_layer = BatchNormalization()

    def transform(self, Input):
        with tf.variable_scope(self.name):
            hidden = tf.identity(Input)
            for hidden_layer in self.hidden_layers:
                hidden = hidden_layer(hidden)

            mu = self.mu_layer(hidden)
            if self.use_residual:
                mu_shape = tf.shape(mu)
                mu_reshaped = tf.reshape(mu, [-1, self.Dout])
                mu = tf.reshape(self.batch_norm_layer(mu_reshaped, training=self.training), mu_shape)
                mu += Input[..., :self.Dout]

        return mu
