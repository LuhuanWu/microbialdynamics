from src.transformation.ilr_clv_utils import get_psi, inverse_ilr_transform
import tensorflow as tf


class inv_ilr_transformation(object):
    # base class for transformation
    def __init__(self, theta, ilr_clv=None):
        self.psi = get_psi(theta)
        self.ilr_clv = ilr_clv

    def transform(self, x):
        p = inverse_ilr_transform(x, self.psi)
        if self.ilr_clv is not None and self.ilr_clv.inference_schedule == "bottom_up":
            # self-normalize inside each tree
            p = self.ilr_clv.get_normalized_p_t(p)
        logp = tf.log(p)
        return logp
