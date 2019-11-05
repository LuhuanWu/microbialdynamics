import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization

from src.transformation.base import transformation


class MLP_transformation(transformation):
    def __init__(self, Dhs, Dout,
                 use_residual=False,
                 name="MLP_transformation"):
        self.Dhs = Dhs
        self.Dout = Dout

        self.use_residual = use_residual

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

            self.mu_layer = Dense(self.Dout,
                                  activation="linear",
                                  kernel_initializer="he_normal",
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
                mu = tf.reshape(self.batch_norm_layer(mu_reshaped), mu_shape) + Input[..., :self.Dout]

        return mu
