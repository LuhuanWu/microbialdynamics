import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization

from src.transformation.base import transformation


class MLP_transformation(transformation):
    def __init__(self, Dhs, Dout,
                 batch_norm=False,
                 name="MLP_transformation"):
        self.Dhs = Dhs
        self.Dout = Dout
        self.batch_norm = batch_norm

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

            if self.batch_norm:
                self.batch_norm_layer = BatchNormalization()

    def transform(self, Input, **kwargs):
        with tf.variable_scope(self.name):
            hidden = tf.identity(Input)
            for hidden_layer in self.hidden_layers:
                hidden = hidden_layer(hidden)

            mu = self.mu_layer(hidden)
            if self.batch_norm:
                mu = self.batch_norm_layer(mu)

        return mu
