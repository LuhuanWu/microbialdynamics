import tensorflow as tf

from src.transformation.base import transformation

class clv_transformation(transformation):
    def __init__(self, Dx, Dev):
        self.Dx = Dx
        self.Dev = Dev

        self.A = tf.Variable(tf.zeros((self.Dx + 1, self.Dx)))
        self.g = tf.Variable(tf.zeros((self.Dx,)))
        self.Wg = tf.Variable(tf.zeros((self.Dev, self.Dx)))

    def transform(self, Input):
        """
        :param Input: (n_particles, batch_size, Dx + Dev)
        :param Dx: dimension of hidden space
        :return: output: (n_particles, batch_size, Dx)
        """
        # x_t + g_t + v_t * Wg + p_t * A

        A, g, Wg = self.A, self.g, self.Wg
        Dx = self.Dx

        shape = Input.shape.as_list()
        if len(shape) > 3:
            batch_size, DxpDev = shape[-2], shape[-1]
            shape[-1] = Dx
            Input = tf.reshape(Input, (-1, batch_size, DxpDev))
        assert Dx > 0

        x = Input[..., 0:Dx]  # (n_particles, batch_size, Dx)
        v = Input[0, 0:1, Dx:] # (1, Dev)
        v_size = v.shape[-1]

        zeros = tf.zeros_like(x[..., 0:1])
        p = tf.concat([x, zeros], axis=-1)
        p = tf.nn.softmax(p, axis=-1)  # (n_particles, batch_size, Dx + 1)

        if v_size > 0:
            # Wg shape (Dev, Dx)
            Wgv = batch_matmul(v, Wg)  # (n_particles, batch_size, Dx)

            output = x + g + Wgv + batch_matmul(p, A)
        else:
            output = x + g + batch_matmul(p, A)

        if len(shape) > 3:
            output = tf.reshape(output, shape)
        return output


def batch_matmul(Input, A):
    """
    Input * A
    :param A: (m, n)
    :param Input: (..., m)
    :return: (..., n)
    """

    Input_shape = Input.shape.as_list()  # (..., m)
    output_shape = tf.unstack(tf.shape(Input))
    output_shape[-1] = tf.shape(A)[-1]  # output_shape = (..., n)
    output_shape = tf.stack(output_shape)

    Input_reshaped = tf.reshape(Input, [-1, Input_shape[-1]])
    output_reshaped = tf.matmul(Input_reshaped, A)

    output = tf.reshape(output_reshaped, output_shape)

    return output