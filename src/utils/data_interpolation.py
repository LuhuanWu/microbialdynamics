import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt

def interpolate_data(hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, FLAGS):
    interpolated_hidden_train = []
    interpolated_hidden_test = []
    interpolated_obs_train = []
    interpolated_obs_test = []
    interpolated_input_train = []
    interpolated_input_test = []
    mask_train = []
    mask_test = []
    time_interval_train = []
    time_interval_test = []

    for hidden, obs, input in zip(hidden_train, obs_train, input_train):
        hidden, obs, input, mask, time_interval = interpolate_datapoint(hidden, obs, input, FLAGS)
        interpolated_hidden_train.append(hidden)
        interpolated_obs_train.append(obs)
        interpolated_input_train.append(input)
        mask_train.append(mask)
        time_interval_train.append(time_interval)

    for hidden, obs, input in zip(hidden_test, obs_test, input_test):
        hidden, obs, input, mask, time_interval = interpolate_datapoint(hidden, obs, input, FLAGS)
        interpolated_hidden_test.append(hidden)
        interpolated_obs_test.append(obs)
        interpolated_input_test.append(input)
        mask_test.append(mask)
        time_interval_test.append(time_interval)

    return interpolated_hidden_train, interpolated_hidden_test, \
           interpolated_obs_train, interpolated_obs_test, \
           interpolated_input_train, interpolated_input_test, \
           mask_train, mask_test, time_interval_train, time_interval_test


def interpolate_datapoint(hidden, obs, input, FLAGS):
    """

    :param hidden: (n_obs, Dx)
    :param obs: (n_obs, Dy + 1), [:, 0] records t of all obs
    :param input: (n_inputs, Dy + 1], [:, 0] records t of all inputs
    :return:
    hidden: (time, Dx)
    obs: (time, Dy)
    interpolated_input: (time, Dv)
    mask: (time, )
    time_interval: (time, )
    """
    days = obs[:, 0].astype(int)
    time = days[-1] - days[0] + 1

    mask = np.ones((time, ), dtype=bool)
    time_interval = np.zeros((time, ))

    i = 0
    for t in np.arange(days[0], days[-1] + 1):
        idx = t - days[0]
        if t == days[i]:
            i = i + 1
            time_interval[idx] = 0
        else:
            mask[t - days[0]] = False
            time_interval[idx] = time_interval[idx - 1] + 1

    # hidden
    hidden = np.zeros((time, hidden.shape[1]))

    # obs
    Dy = obs.shape[1] - 1
    i = 0
    if FLAGS.dirichlet_emission:
        interpoated_obs = np.zeros((time, Dy))
        last_valid_value = np.ones(Dy) / Dy
        for t in np.arange(days[0], days[-1] + 1):
            if t == days[i]:
                smoothed_obs = obs[i, 1:]
                smoothed_obs = smoothed_obs * (1 - 1e-6) + 1e-6 / Dy
                interpoated_obs[t - days[0]] = smoothed_obs
                last_valid_value = smoothed_obs
                i += 1
            else:
                interpoated_obs[t - days[0]] = last_valid_value
    else:
        X = np.atleast_2d(days).T
        y = obs[:, 1:]

        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        noise = 1e-2
        gp = GaussianProcessRegressor(kernel=kernel, alpha=noise ** 2, n_restarts_optimizer=10)
        gp.fit(X, y)

        X_pred = np.atleast_2d(np.arange(days[0], days[-1] + 1)).T
        interpoated_obs, sigma = gp.predict(X_pred, return_std=True)

        # plt.figure()
        # plt.plot(X_pred, interpoated_obs, 'b-', label='Prediction')
        # plt.fill(np.concatenate([X_pred, X_pred[::-1]]),
        #          np.concatenate([interpoated_obs - 1.9600 * sigma,
        #                          (interpoated_obs + 1.9600 * sigma)[::-1]]),
        #          alpha=.5, fc='b', ec='None', label='95% confidence interval')
        # plt.xlabel('$x$')
        # plt.ylabel('$y$')
        # plt.legend(loc='upper left')
        # plt.show()

    # input
    Dv = input.shape[1] - 1
    interpoated_input = np.zeros((time, Dv))
    for day_input in input:
        day = int(day_input[0])
        if days[0] <= day <= days[-1]:
            interpoated_input[day - days[0]] = day_input[1:]

    return hidden, interpoated_obs, interpoated_input, mask, time_interval





