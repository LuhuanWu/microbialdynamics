import tensorflow as tf


class identity_transformation(object):
    # base class for transformation
    def __init__(self, params=None):
        self.params = params

    def transform(self, x):
        return x
