import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def interpolate_data(hidden_train, hidden_test, obs_train, obs_test, input_train, input_test,
                     extra_inputs_train, extra_inputs_test, use_gp, interpolation_data=None):
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
    interpolated_extra_inputs_train = []
    interpolated_extra_inputs_test = []

    if interpolation_data is not None:
        # print(len(interpolation_data), len(obs_train))
        assert len(interpolation_data) == len(obs_train)
    else:
        interpolation_data = [None] * len(obs_train)

    for hidden, obs, input, extra_inputs, interpolation in \
            zip(hidden_train, obs_train, input_train, extra_inputs_train, interpolation_data):
        hidden, obs, input, mask, time_interval, extra_inputs = \
            interpolate_datapoint(hidden, obs, input, extra_inputs, use_gp, interpolation)
        interpolated_hidden_train.append(hidden)
        interpolated_obs_train.append(obs)
        interpolated_input_train.append(input)
        mask_train.append(mask)
        time_interval_train.append(time_interval)
        interpolated_extra_inputs_train.append(extra_inputs)

    for hidden, obs, input, extra_inputs in zip(hidden_test, obs_test, input_test, extra_inputs_test):
        hidden, obs, input, mask, time_interval, extra_inputs = interpolate_datapoint(hidden, obs, input, extra_inputs, use_gp)
        interpolated_hidden_test.append(hidden)
        interpolated_obs_test.append(obs)
        interpolated_input_test.append(input)
        mask_test.append(mask)
        time_interval_test.append(time_interval)
        interpolated_extra_inputs_test.append(extra_inputs)

    return interpolated_hidden_train, interpolated_hidden_test, \
           interpolated_obs_train, interpolated_obs_test, \
           interpolated_input_train, interpolated_input_test, \
           mask_train, mask_test, time_interval_train, time_interval_test, \
           interpolated_extra_inputs_train, interpolated_extra_inputs_test


def interpolate_datapoint(hidden, obs, input, extra_inputs, use_gp,
                          interpolation=None, pseudo_count=1, pseudo_percentage=1e-6):
    """

    :param hidden: (n_obs, Dx)
    :param obs: (n_obs, Dy + 1), [:, 0] records t of all obs
    :param input: (n_inputs, Dy + 1], [:, 0] records t of all inputs
    :param extra_inputs: (n_obs_inputs, )
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

    if interpolation is not None:
        interpolation = np.round(interpolation).astype(int) + pseudo_count

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

    if use_gp:
        X = np.atleast_2d(days).T
        y = obs[:, 1:]

        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        noise = 1e-2
        gp = GaussianProcessRegressor(kernel=kernel, alpha=noise ** 2, n_restarts_optimizer=10)
        gp.fit(X, y)

        X_pred = np.atleast_2d(np.arange(days[0], days[-1] + 1)).T
        interpolated_obs, sigma = gp.predict(X_pred, return_std=True)
    else:
        interpolated_obs = np.zeros((time, Dy))
        last_valid_value = np.ones(Dy) / Dy
        i = 0
        for t in np.arange(days[0], days[-1] + 1):
            if t == days[i]:
                smoothed_obs = obs[i, 1:]
                if np.abs(np.sum(smoothed_obs) - 1) < 1e-6: # if use percentage data & dirichlet emission
                    smoothed_obs = smoothed_obs * (1 - pseudo_percentage) + pseudo_percentage / Dy
                else:
                    smoothed_obs += pseudo_count
                interpolated_obs[t - days[0]] = smoothed_obs
                last_valid_value = smoothed_obs
                i += 1
            else:
                interpolated_obs[t - days[0]] = last_valid_value
        if interpolation is not None:
            assert interpolation.shape == (time, Dy)
            for i, (m, obs, iobs) in enumerate(zip(mask, interpolated_obs, interpolation)):
                interpolated_obs[i] = obs if m else iobs

    # input
    Dv = input.shape[1] - 1
    interpolated_input = np.zeros((time, Dv))
    for day_input in input:
        day = int(day_input[0])
        if days[0] <= day <= days[-1]:
            interpolated_input[day - days[0]] = day_input[1:]

    # extra_inputs
    interpolated_extra_inputs = np.zeros(time)

    if extra_inputs is not None:
        last_valid_value = extra_inputs[0] + Dy * pseudo_count
        i = 0
        for t in np.arange(days[0], days[-1] + 1):
            if t == days[i]:
                interpolated_extra_inputs[t - days[0]] = extra_inputs[i] + Dy * pseudo_count
                last_valid_value = extra_inputs[i] + Dy * pseudo_count
                i += 1
            else:
                interpolated_extra_inputs[t - days[0]] = last_valid_value
    if interpolation is not None:
        for i, (m, ei, iobs) in enumerate(zip(mask, interpolated_extra_inputs, interpolation)):
            interpolated_extra_inputs[i] = ei if m else np.sum(iobs)

    return hidden, interpolated_obs, interpolated_input, mask, time_interval, interpolated_extra_inputs






