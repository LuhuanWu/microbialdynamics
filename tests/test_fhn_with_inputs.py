import numpy as np

from src.transformation.fhn import fhn_transformation
from src.transformation.linear import linear_transformation
from src.distribution.dirac_delta import dirac_delta
from src.distribution.mvn import mvn
from src.utils.data_generator import generate_hidden_obs, generate_dataset

import joblib
import os


########### test generate hidden and observation of fhn with inputs #################
"""
time = 10

Dx = 2
Dy = 1
Dv = 1

x_0 = np.random.randn(Dx)  #
inputs = np.random.rand(time, Dv) * 3

# specifying hidden dynamics
a, b, c, dt = 1.0, 0.95, 0.05, 0.15
f_params = (a, b, c, dt)
f_tran = fhn_transformation(f_params)
f = dirac_delta(f_tran)

# specifying observation distribution
g_params = np.array([[1.0, 0.0]])
g_cov = 0.01 * np.eye(Dy)
g_tran = linear_transformation(g_params)
g = mvn(g_tran, g_cov)

out = generate_hidden_obs(time, Dx, Dy, x_0, f, g, inputs=inputs, Dv=Dv)
print(len(out))

print(out[0].shape)

print(out[1].shape)
"""

############# test generate datasets ########################


n_train = 200
n_test = 40
time = 200
Dx = 2
Dy = 1

Dv = 1

inputs = np.random.rand(n_train+n_test, time, Dv) * 3

hidden_train, hidden_test, obs_train, obs_test = generate_dataset(n_train, n_test, time,
                                                     model="fhn", Dx=Dx, Dy=1,
                                                     f=None, g=None,
                                                     x_0_in=None, lb=-2.5, ub=2.5, inputs=inputs, Dv=1)

assert hidden_train.shape == (n_train, time, Dx)
assert hidden_test.shape == (n_test, time, Dx)
assert obs_train.shape == (n_train, time, Dy)
assert obs_test.shape == (n_test, time, Dy)

inputs_train = inputs[:n_train]
inputs_test = inputs[n_train:]

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(repo_dir, "data/fhn_with_inputs")
print(data_dir)
joblib.dump((hidden_train, hidden_test, obs_train, obs_test, inputs_train, inputs_test), data_dir)



