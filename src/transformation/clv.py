import tensorflow as tf
import pickle

from src.transformation.base import transformation

class clv_transformation(transformation):
    def __init__(self, Dx, Dev, beta_constant=True, clv_in_alr=True, data_dir=None):
        self.Dx = Dx
        self.Dev = Dev
        self.clv_in_alr = clv_in_alr

        if data_dir is not None:
            with open(data_dir, "rb") as f:
                d = pickle.load(f)
            A_init_val = tf.constant(d['A'], dtype=tf.float32)
            g_init_val = tf.constant(d['g'], dtype=tf.float32)
            Wv_init_val = tf.constant(d['Wv'].T, dtype=tf.float32)

        hidden_dim = self.Dx + 1 if clv_in_alr else self.Dx
        if data_dir is None or d['A'].shape[0] != hidden_dim:
            A_init_val = tf.zeros((hidden_dim, hidden_dim))
            g_init_val = tf.zeros((hidden_dim,))
            Wv_init_val = tf.zeros((self.Dev, hidden_dim))
        self.A_var = tf.Variable(A_init_val)
        self.g_var = tf.Variable(g_init_val)
        self.Wv_var = tf.Variable(Wv_init_val)

        if not beta_constant:
            upper_triangle = tf.linalg.band_part(self.A_var, 0, -1)     # including diagonal
            upper_triangle = -tf.nn.softplus(upper_triangle)
            upper_triangle = tf.linalg.band_part(upper_triangle, 0, -1)
            lower_triangle = tf.linalg.band_part(self.A_var, -1, 0)     # including diagonal
            diagonal = -tf.nn.softplus(tf.linalg.diag_part(self.A_var))
            A_tmp = upper_triangle + lower_triangle
            self.A = tf.linalg.set_diag(A_tmp, diagonal)
        else:
            self_interaction = -tf.nn.softplus(tf.linalg.diag_part(self.A_var))
            self.A = tf.linalg.set_diag(self.A_var, self_interaction)   # self-interaction should be negative
        self.g = tf.nn.softplus(self.g_var)                             # growth should be positive
        self.Wv = self.Wv_var

    def transform(self, Input):
        """
        :param Input: (n_particles, batch_size, Dx + Dev)
        :param Dx: dimension of hidden space
        :return: output: (n_particles, batch_size, Dx)
        """
        # x_t + g_t + v_t * Wv + p_t * A

        A, g, Wv = self.A, self.g, self.Wv
        Dx = self.Dx

        """
        shape = Input.shape.as_list()
        if len(shape) > 3:
            batch_size, DxpDev = shape[-2], shape[-1]
            shape[-1] = Dx
            Input = tf.reshape(Input, (-1, batch_size, DxpDev))
        assert Dx > 0
        """

        x = Input[..., 0:Dx]  # (n_particles, batch_size, Dx)
        v = Input[0, 0:1, Dx:] # (1, Dev)
        v_size = v.shape[-1]

        if self.clv_in_alr:
            zeros = tf.zeros_like(x[..., 0:1])
            x = tf.concat([x, zeros], axis=-1)
        p = tf.nn.softmax(x, axis=-1)  # (n_particles, batch_size, Dx + 1)

        # (..., Dx+1, 1) * (Dx+1, Dx)
        pA = tf.reduce_sum(p[..., None] * A, axis=-2) # (..., Dx)
        if v_size > 0:
            # Wv shape (Dev, Dx)
            Wvv = batch_matmul(v, Wv)  # (n_particles, batch_size, Dx)
            output = x + g + Wvv + pA
        else:
            output = x + g + pA

        if self.clv_in_alr:
            output = output[..., :-1] - output[..., -1:]

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


def test_batch_matmul():
    import numpy as np

    A = tf.reshape(tf.range(12), (4,3))
    Input = tf.reshape(tf.range(720), (2,5,6,3,4))

    out_1 = batch_matmul(Input, A)
    out_2 = tf.reduce_sum(Input[..., None]*A, axis=-2)

    sess = tf.Session()
    out_1_np = sess.run(out_1)
    out_2_np = sess.run(out_2)

    assert np.all(out_1_np, out_2_np)

    A = np.reshape(np.arange(12), (4,3))
    Input = np.reshape(np.arange(720), (2,5,6,3,4))

    out = np.matmul(Input, A)
    assert np.all(out, out_1_np)