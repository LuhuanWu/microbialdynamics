import numpy as np
import tensorflow as tf

from src.transformation.base import transformation
from src.transformation.clv import batch_matmul


class clv_original_transformation(transformation):

    def transform(self, Input, Dx=-1):
        """

        :param Input: (n_particles, batch_size, Dx + Dev)
        :param Dx: dimension of hidden space
        :return: output: (n_particles, batch_size, Dx)
        """
        # x_t + g_t + v_t * Wg + p_t * A

        A, g, Wg = self.params
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

            output = x + g + Wgv + batch_matmul(p, A)
        else:
            output = x + g + batch_matmul(p, A)

        return output
