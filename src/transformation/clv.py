import tensorflow as tf

from src.transformation.base import transformation

class clv_transformation(transformation):
    def __init__(self, Dx, Dv, reg_coef=1.0, annealing_frac=1.0):
        self.Dx = Dx
        self.Dv = Dv
        self.reg_coef = reg_coef
        self.annealing_frac = annealing_frac

        A_init_val = tf.zeros((Dx + 1, Dx))
        g_init_val = tf.zeros((Dx,))
        Wv_init_val = tf.zeros((self.Dv, Dx))
        self.A_var = tf.Variable(A_init_val)
        self.g_var = tf.Variable(g_init_val)
        self.Wv_var = tf.Variable(Wv_init_val)

        self.A = self.A_var
        self.g = self.g_var
        self.Wv = self.Wv_var

    def transform(self, Input):
        """
        :param Input: (n_particles, batch_size, Dx + Dv)
        :param Dx: dimension of hidden space
        :return: output: (n_particles, batch_size, Dx)
        """
        # x_t + g_t + v_t * Wv + p_t * A

        A, g, Wv = self.A, self.g, self.Wv
        Dx = self.Dx

        x = Input[..., 0:Dx]  # (n_particles, batch_size, Dx)
        v = Input[0, 0:1, Dx:]  # (1, Dv)
        v_size = v.shape[-1]

        zeros = tf.zeros_like(x[..., 0:1])
        x_ = tf.concat([x, zeros], axis=-1)
        p = tf.nn.softmax(x_, axis=-1)  # (n_particles, batch_size, Dx + 1)

        # (..., Dx+1, 1) * (Dx+1, Dx)
        pA = tf.reduce_sum(p[..., None] * A, axis=-2)  # (..., Dx)
        if v_size > 0:
            # Wv shape (Dv, Dx)
            Wvv = tf.reduce_sum(v[..., None] * Wv, axis=-2)
            output = x + g + Wvv + pA
        else:
            output = x + g + pA

        return output

    def regularization_loss(self):
        L2 = tf.reduce_sum(self.A_var ** 2) + tf.reduce_sum(self.g_var ** 2) + tf.reduce_sum(self.Wv_var ** 2)
        return L2 * self.reg_coef * self.annealing_frac
