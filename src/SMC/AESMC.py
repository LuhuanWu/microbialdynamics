import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from src.SMC.SVO import SVO


class AESMC(SVO):
    def __init__(self, model, FLAGS, name="log_ZSMC"):
        SVO.__init__(self, model, FLAGS, name)
        self.smooth_obs = False
