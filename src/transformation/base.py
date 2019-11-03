from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class transformation(object):
    # base class for transformation
    def __init__(self, params=None):
        self.params = params

    @abstractmethod
    def transform(self, X_prev):
        pass


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
