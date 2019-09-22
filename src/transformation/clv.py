import numpy as np
import tensorflow as tf

from src.transformation.base import transformation


class clv_transformation(transformation):

    def transform(self, Input, Dx=-1):
        """

        :param Input: (n_particles, batch_size, Dx + Dev)
        :param Dx: dimension of hidden space
        :return: output: (n_particles, batch_size, Dx)
        """
        # x_t + g_t + v_t * Wg + p_t * (A+ A(v_t)) where A(v_t) = W2 * (v_t * W1)

        A, g, Wg, W1, W2 = self.params
        # A (Dx + 1, Dx)
        # g (Dx, )
        # Wg (Dev, Dx)
        # W1 (Dev, Dx)
        # W2 (Dx+1, 1)

        assert len(Input.shape) == 3
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

            # (1, Dev) * (Dev, Dx) --> (1, Dx)
            Aofv = tf.matmul(v, W1)
            # (Dx + 1, 1) * (1, Dx) --> (Dx+1, Dx)
            Aofv = tf.matmul(W2, Aofv)

            output = x + g + Wgv + batch_matmul(p, A + Aofv)
        else:
            output = x + g + batch_matmul(p, A)

        return output


def batch_matmul(Input, A):
    """
    Input * A
    :param A: (m, n)
    :param Input: (..., m)
    :return: (..., n)
    """
    # some adhoc way to cope with this: assume batch_size = 1
    Input_shape = Input.shape.as_list()  # (..., m)

    output_shape = list(Input_shape)
    output_shape[-1] = A.shape.as_list()[-1]  # output_shape = (..., n)

    num_None = 0
    for i in range(len(output_shape)):
        if output_shape[i] is None:
            output_shape[i] = -1
            num_None += 1
    if num_None > 1:
        raise ValueError("Shape error! Can have most one None in shape.")

    # (-1, m) * (m, n) --> (-1, n)
    Input_reshaped = tf.reshape(Input, [-1, Input_shape[-1]])
    output_reshaped = tf.matmul(Input_reshaped, A)

    output = tf.reshape(output_reshaped, output_shape)

    return output

