import numpy as np
import tensorflow as tf

from src.transformation.base import transformation


class clv_transformation(transformation):

    def transform(self, Input, Dx=-1):
        # Input shape: (n_particles, batch_size, Dx + Dev)

        # x_t + g_t + Wg v_t + (A+Wa v_t) * p_t

        A, g, Wa, Wg = self.params
        assert Dx > 0

        x = Input[..., 0:Dx]
        v = Input[..., Dx:]  # (..., Dev)

        zeros = tf.zeros_like(x[..., 0:1])
        p = tf.concat([x, zeros], axis=-1)
        p = tf.nn.softmax(p, axis=-1)  # (n_particles, batch_size, Dx + 1)

        # Wg shape (Dx, Dev).
        Wgv = batch_matmul(Wg, v)  # (..., Dx)

        # Wa shape (Dx, Dev)
        Wav = batch_matmul(Wa, v) # (..., Dx)

        Wav_by_p = Wav * tf.reduce_sum(p, axis=-1, keepdims=True)  # (...., Dx)

        # A shape: (Dx, Dx + 1)
        output = x + g + Wgv + batch_matmul(A, p) + Wav_by_p

        return output


def batch_matmul(A, Input):
    """

    :param A: (n, m)
    :param Input: (..., m)
    :return:
    """
    # some adhoc way to cope with this: assume batch_size = 1
    Input_shape = Input.shape.as_list()  # (..., m)

    output_shape = list(Input_shape)
    output_shape[-1] = A.shape.as_list()[0]  # (..., n)
    #output_shape = (-1, 1, A.shape.as_list()[0])

    num_None = 0
    for i in range(len(output_shape)):
        if output_shape[i] == None:
            output_shape[i] = -1
            num_None += 1
    if num_None > 1:
        raise ValueError("Shape error! Could have have most one None in shape.")

    # (-1, m) * (m, n) --> (-1, n)
    Input_reshaped = tf.reshape(Input, [-1, Input_shape[-1]])
    output_reshaped = tf.matmul(Input_reshaped, A, transpose_b=True)

    output = tf.reshape(output_reshaped, output_shape)

    return output

