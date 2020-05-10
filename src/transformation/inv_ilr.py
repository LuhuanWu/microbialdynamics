from src.transformation.ilr_clv_utils import get_psi, inverse_ilr_transform
import tensorflow as tf


class inv_ilr_transformation(object):
    # base class for transformation
    def __init__(self, theta):
        self.psi = get_psi(theta)

    def transform(self, x):
        p = inverse_ilr_transform(x, self.psi)
        logp = tf.log(p)
        return logp
