import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import activations

from src.transformation.base import transformation, xavier_init


class ExpandedMLPTransformation(transformation):
    """
    This is equivalent to #batch_size MLPs, each takes in (..., d_in) and outputs (..., d_out).
    In general, it takes in (..., batch_size, d_in) and outputs (..., batch_size, d_out)
    """
    def __init__(self, batch_size, Dhs, Dout, name="MLP_transformation"):
        """Dhs is shared for all batch_size"""
        self.batch_size = batch_size
        self.Dhs = Dhs
        self.Dout = Dout

        self.name = name
        self.init_FFN()

    def init_FFN(self):
        with tf.variable_scope(self.name):
            self.hidden_layers = []

            for i, Dh in enumerate(self.Dhs):
                self.hidden_layers.append(
                    ExpandedDense(batch_size=self.batch_size,
                                  d_out=Dh,
                                  activation="relu",
                                  name="hidden_{}".format(i))
                )

            self.mu_layer = ExpandedDense(batch_size=self.batch_size,
                                          d_out=self.Dout,
                                          activation="linear",
                                          name="mu_layer")

    def transform(self, Input):
        """

        :param Input: (..., batch_size, d_in)
        :return: (..., batch_size, d_out)
        """
        with tf.variable_scope(self.name):
            hidden = tf.identity(Input)
            for hidden_layer in self.hidden_layers:
                hidden = hidden_layer(hidden)
            mu = self.mu_layer(hidden)
        return mu


class ExpandedDense():
    def __init__(self, batch_size, d_out, activation=None, use_bias=True, name=None):
        self.batch_size = batch_size
        self.d_out = d_out
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.name = name

        self.built = False

    def build(self, inputs):
        assert not self.built
        self.d_in = inputs.shape.as_list()[-1]
        with tf.variable_scope(self.name):
            self.weights = tf.Variable(xavier_init(fan_in=self.d_in, fan_out=self.d_out, batch_size=self.batch_size)) # (batch_size, d_in, d_out)
            self.bias = tf.Variable(tf.zeros((self.batch_size, self.d_out)))# (batch_size, d_out)
        self.built= True

    def __call__(self, inputs):
        # inputs: (..., batch_size, d_in)
        rank = len(inputs.shape)
        assert rank >= 2, inputs.shape

        if not self.built:
            self.build(inputs)

        # (..., bs, d_in, 1) * (bs, d_in, d_out)
        out = tf.reduce_sum(inputs[..., None]*self.weights, axis=-2) # (..., bs, d_out)
        if self.use_bias:
            out = out + self.bias
        out = self.activation(out)
        return out


class ExpandedMLPTransformation_v2(transformation):
    """
    This is equivalent to applying a batch of MLP to the same input
    Input shape (..., d_in). outputshape (..., batch_size, d_out).
    """
    def __init__(self, batch_size, Dhs, Dout, name="MLP_transformation"):
        """Dhs is shared for all batch_size"""
        self.batch_size = batch_size
        self.Dhs = Dhs
        self.Dout = Dout

        self.name = name
        self.init_FFN()

    def init_FFN(self):
        with tf.variable_scope(self.name):
            self.hidden_layers = []
            for i, Dh in enumerate(self.Dhs):
                if i == 0:
                    self.hidden_layers.append(
                        ExpandedDense_v2(batch_size=self.batch_size,
                                      d_out=Dh,
                                      activation="relu",
                                      name="hidden_{}".format(i)))
                else:
                    self.hidden_layers.append(
                        ExpandedDense(batch_size=self.batch_size,
                                         d_out=Dh,
                                         activation="relu",
                                         name="hidden_{}".format(i)))

            self.mu_layer = ExpandedDense(batch_size=self.batch_size,
                                          d_out=self.Dout,
                                          activation="linear",
                                          name="mu_layer")

    def transform(self, Input):
        """

        :param Input: (..., d_in)
        :return: (..., batch_size, d_out)
        """
        with tf.variable_scope(self.name):
            hidden = tf.identity(Input)
            for hidden_layer in self.hidden_layers:
                hidden = hidden_layer(hidden)
            mu = self.mu_layer(hidden)
        return mu


class ExpandedDense_v2():
    def __init__(self, batch_size, d_out, activation=None, use_bias=True, name=None):
        self.batch_size = batch_size
        self.d_out = d_out
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.name = name

        self.built = False

    def build(self, inputs):
        assert not self.built
        self.d_in = inputs.shape.as_list()[-1]
        with tf.variable_scope(self.name):
            self.weights = tf.Variable(
                xavier_init(fan_in=self.d_in, fan_out=self.d_out, batch_size=self.batch_size))  # (batch_size, d_in, d_out)
            self.bias = tf.Variable(tf.zeros((self.batch_size, self.d_out)))  # (batch_size, d_out)
        self.built = True

    def __call__(self, inputs):
        # inputs: (..., d_in)
        rank = len(inputs.shape)

        if not self.built:
            self.build(inputs)

        # (...,  1, d_in, 1) * (bs, d_in, d_out)
        inputs = tf.expand_dims(inputs, axis=-2)
        out = tf.reduce_sum(inputs[..., None]*self.weights, axis=-2) # (..., bs, d_out)
        if self.use_bias:
            out = out + self.bias
        out = self.activation(out)
        return out
