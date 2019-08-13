from src.distribution.base import distribution


# np ver, just used in sampler, so no need to implement log_prob
class dirac_delta(distribution):
    """
    multivariate dirac_delta distribution,
    used for sampling from fhn, lorenz model without noise
    """
    def __init__(self, transformation):
        self.transformation = transformation

    def sample(self, Input, ext_inputs=None):
        if ext_inputs is None:
            return self.transformation.transform(Input)
        return self.transformation.transform(Input, ext_inputs)


