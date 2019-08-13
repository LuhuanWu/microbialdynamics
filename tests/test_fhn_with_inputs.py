import numpy as np

from src.transformation.fhn import fhn_transformation
from src.transformation.linear import linear_transformation
from src.distribution.dirac_delta import dirac_delta
from src.distribution.mvn import mvn
from src.utils.data_generator import generate_hidden_obs

########### test generate hidden and observation of fhn with inputs #################

time = 10

Dx = 2
Dy = 1

x_0 = np.random.randn(Dx)  #


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

out = generate_hidden_obs(time, Dx, Dy, x_0, f, g, inputs=None, Dv=1)
print(len(out))

print(out[0].shape)

print(out[1].shape)

