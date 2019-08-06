import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt

def interpolate_data(hidden_train, hidden_test, obs_train, obs_test, input_train, input_test):
    interpolated_hidden_train = []
    interpolated_hidden_test = []
    interpolated_obs_train = []
    interpolated_obs_test = []
    interpolated_input_train = []
    interpolated_input_test = []

    for hidden, obs, input in zip(hidden_train, obs_train, input_train):
        hidden, obs, input = interpolate_datapoint(hidden, obs, input)
        interpolated_hidden_train.append(hidden)
        interpolated_obs_train.append(obs)
        interpolated_input_train.append(input)

    for hidden, obs, input in zip(hidden_test, obs_test, input_test):
        hidden, obs, input = interpolate_datapoint(hidden, obs, input)
        interpolated_hidden_test.append(hidden)
        interpolated_obs_test.append(obs)
        interpolated_input_test.append(input)

    return interpolated_hidden_train, interpolated_hidden_test, \
           interpolated_obs_train, interpolated_obs_test, \
           interpolated_input_train, interpolated_input_test

def interpolate_datapoint(hidden, obs, input):
    """

    :param hidden: (n_obs, Dx)
    :param obs: (n_obs, Dy + 1), [:, 0] records t of all obs
    :param input: (n_inputs, Dy + 1], [:, 0] records t of all inputs
    :return:
    """
    days = obs[:, 0].astype(int)
    time = days[-1] - days[0] + 1
    # hidden
    hidden = np.zeros((time, hidden.shape[1]))

    # obs
    X = np.atleast_2d(days).T
    y = obs[:, 1:]

    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    noise = 1e-2
    gp = GaussianProcessRegressor(kernel=kernel, alpha=noise ** 2, n_restarts_optimizer=10)
    gp.fit(X, y)

    X_pred = np.atleast_2d(np.arange(days[0], days[-1] + 1)).T
    obs, sigma = gp.predict(X_pred, return_std=True)

    # plt.figure()
    # plt.plot(X_pred, obs, 'b-', label='Prediction')
    # plt.fill(np.concatenate([X_pred, X_pred[::-1]]),
    #          np.concatenate([obs - 1.9600 * sigma,
    #                          (obs + 1.9600 * sigma)[::-1]]),
    #          alpha=.5, fc='b', ec='None', label='95% confidence interval')
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.legend(loc='upper left')
    # plt.show()

    # input
    Dv = input.shape[1] - 1
    input = np.zeros((time, Dv))
    for day_input in input:
        day = int(day_input[0])
        if days[0] <= day <= days[-1]:
            input[day - days[0]] = day_input[1:]

    return hidden, obs, input





