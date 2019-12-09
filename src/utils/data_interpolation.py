import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.misc import logsumexp


def interpolate_data(hidden_train, hidden_test, obs_train, obs_test, input_train, input_test,
                     extra_inputs_train, extra_inputs_test, interpolation_type=None, interpolation_data=None,
                     pseudo_count=1, pseudo_percentage=1e-6):
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
        interpolation_data_train = interpolation_data["train"]
        interpolation_data_test = interpolation_data["test"]
        assert len(interpolation_data_train) == len(obs_train)
        assert len(interpolation_data_test) == len(obs_test)
    else:
        interpolation_data_train = [None] * len(obs_train)
        interpolation_data_test = [None] * len(obs_test)

    for hidden, obs, input, extra_inputs, interpolation in \
            zip(hidden_train, obs_train, input_train, extra_inputs_train, interpolation_data_train):
        hidden, obs, input, mask, time_interval, extra_inputs = \
            interpolate_datapoint(hidden, obs, input, extra_inputs, interpolation_type=interpolation_type,
                                  interpolation=interpolation,
                                  pseudo_count=pseudo_count, pseudo_percentage=pseudo_percentage)

        interpolated_hidden_train.append(hidden)
        interpolated_obs_train.append(obs)
        interpolated_input_train.append(input)
        mask_train.append(mask)
        time_interval_train.append(time_interval)
        interpolated_extra_inputs_train.append(extra_inputs)

    for hidden, obs, input, extra_inputs, interpolation in \
            zip(hidden_test, obs_test, input_test, extra_inputs_test, interpolation_data_test):
        hidden, obs, input, mask, time_interval, extra_inputs = \
            interpolate_datapoint(hidden, obs, input, extra_inputs, interpolation_type=interpolation_type,
                                  interpolation=interpolation,
                                  pseudo_count=pseudo_count, pseudo_percentage=pseudo_percentage)

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


def interpolate_datapoint(hidden, obs, input, extra_inputs, interpolation_type=None,
                          interpolation=None, pseudo_count=1, pseudo_percentage=1e-6):
    """
    :param hidden: ndarray of shape (n_obs, Dx)
    :param obs: ndarray of shape (n_obs, Dy + 1), where obs[:, 0] records day information
    :param input: ndarray of shape (n_inputs, Dy + 1], where input[:, 0] records day information
    :param extra_inputs: nrarray of shape (n_obs_inputs, )
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

    if interpolation_type is not None:
        assert interpolation_type in ["linear_lar", "gp_lar", "gp", "clv"], \
            "interpolation type must be one of linear_lar, gp_lar, gp and clv, " \
            "but receives input as {}".format(interpolation_type)

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
    last_valid_value = np.ones(Dy) / Dy

    if interpolation_type == "gp_lar":
        # lar transformation
        lar = lar_transform_with_pseudopercentage(obs)

        # gp intepolate
        X = np.atleast_2d(days).T
        y = lar

        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        noise = 1e-2
        gp = GaussianProcessRegressor(kernel=kernel, alpha=noise ** 2, n_restarts_optimizer=10)
        gp.fit(X, y)

        X_pred = np.atleast_2d(np.arange(days[0], days[-1] + 1)).T
        interpolated_obs, sigma = gp.predict(X_pred, return_std=True)

        # transform back to count space
        interpolated_obs = inv_lar_transform(interpolated_obs)
        interpolated_obs = np.around(interpolated_obs)
        assert interpolated_obs.shape == (time, Dy)
        interpolated_obs[days - days[0]] = obs[:, 1:]

    elif interpolation_type == "gp":
        # gp intepolate
        X = np.atleast_2d(days).T
        y = obs[:, 1:]

        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        noise = 1e-2
        gp = GaussianProcessRegressor(kernel=kernel, alpha=noise ** 2, n_restarts_optimizer=10)
        gp.fit(X, y)

        X_pred = np.atleast_2d(np.arange(days[0], days[-1] + 1)).T
        interpolated_obs, sigma = gp.predict(X_pred, return_std=True)
        interpolated_obs = np.around(interpolated_obs)

        assert interpolated_obs.shape == (time, Dy)
        interpolated_obs[days - days[0]] = obs[:, 1:]

        assert np.all(interpolated_obs > 0), interpolated_obs

    elif interpolation_type == "linear_lar":
        # lar transformation
        lar = lar_transform_with_pseudopercentage(obs)  # a list, each is of shape (time, Dy-1)

        # linear interpolate
        lar_full = linear_interpolate(lar, days)

        # transform back to count space
        interpolated_obs = inv_lar_transform(lar_full)
        assert interpolated_obs.shape == (time, Dy)
        interpolated_obs[days - days[0]] = obs[:, 1:]

    elif interpolation_type == "clv":
        assert interpolation is not None
        interpolation = np.round(interpolation).astype(int) + pseudo_count
        assert interpolation.shape == (time, Dy), "should be {}, but is {}".format((time, Dy), interpolation.shape)
        interpolated_obs = interpolation
        interpolated_obs[days - days[0]] = obs[:, 1:]

    else:
        interpolated_obs = np.zeros((time, Dy))
        i = 0
        for t in np.arange(days[0], days[-1] + 1):
            if t == days[i]:
                smoothed_obs = obs[i, 1:].copy()
                if np.abs(np.sum(smoothed_obs) - 1) < 1e-6: # if use percentage data & dirichlet emission
                    smoothed_obs = smoothed_obs * (1 - pseudo_percentage) + pseudo_percentage / Dy
                else:
                    smoothed_obs += pseudo_count
                interpolated_obs[t - days[0]] = smoothed_obs
                last_valid_value = smoothed_obs
                i += 1
            else:
                interpolated_obs[t - days[0]] = last_valid_value

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
        if interpolation_type is None:
            last_valid_value = extra_inputs[0] + Dy * pseudo_count
            i = 0
            for t in np.arange(days[0], days[-1] + 1):
                if t == days[i]:
                    interpolated_extra_inputs[t - days[0]] = extra_inputs[i] + Dy * pseudo_count
                    last_valid_value = extra_inputs[i] + Dy * pseudo_count
                    i += 1
                else:
                    interpolated_extra_inputs[t - days[0]] = last_valid_value
        else:
            interpolated_extra_inputs[days - days[0]] = extra_inputs
            for i, (m, ei, iobs) in enumerate(zip(mask, interpolated_extra_inputs, interpolated_obs)):
                interpolated_extra_inputs[i] = ei if m else np.sum(iobs)

    # sanity checks
    if np.abs(np.sum(interpolated_obs, axis=-1)[0] - 1) > 1e-5:
        assert np.allclose(np.sum(interpolated_obs, axis=-1), interpolated_extra_inputs), \
            "sum of counts does not match total counts!, {} != {}".format(np.sum(interpolated_obs, axis=-1),
                                                                                 interpolated_extra_inputs)

    return hidden, interpolated_obs, interpolated_input, mask, time_interval, interpolated_extra_inputs


def lar_transform_with_pseudopercentage(obs):
    """

    :param obs: (time, 1 + Dy)
    :return: lar: (time, Dy-1)
    """
    obs = obs[:, 1:]
    time, Dy = obs.shape
    pseudopercentage = 1e-6

    # first adding pseudo percentage
    percentage = obs / np.sum(obs, axis=-1, keepdims=True)
    percentage = percentage * (1 -  pseudopercentage) + pseudopercentage / Dy

    # lar transfrom
    lar = np.log(percentage[:, :-1]) - np.log(percentage[:, -1:])  # log p_i/p_D

    assert lar.shape == (time, Dy - 1)

    return lar


def inv_lar_transform(lar):
    """
    Only perfrom inverse lar transformation to the interpolated observations
    :param lars: (time, Dy - 1)
    :return: obs: (time, Dy)
    """
    time, Dy_minus_1 = lar.shape
    Dy = Dy_minus_1 + 1

    lar_last = np.zeros((time, 1))
    lar = np.concatenate((lar, lar_last), axis=-1)

    log_percentage = lar - logsumexp(lar, axis=-1, keepdims=True)
    percentage = np.exp(log_percentage)

    count = 1000
    obs = count * percentage
    obs = np.around(obs)

    assert obs.shape == (time, Dy)
    return obs


def linear_interpolate(lars, days):
    """

    :param lars: (time, d)
    :return: lars: (full_time, d)
    """
    time, d = lars.shape
    if time == 1:
        print("no need to interpolate.")
        return lars
    full_time = days[-1] - days[0] + 1

    interpolate_lars = np.zeros((full_time, d))
    last_valid_day_idx = 0
    next_valid_day_idx = 1
    for i, day in enumerate(days[0] + np.arange(full_time)):
        if day == days[last_valid_day_idx]:
            interpolate_lars[i] = lars[last_valid_day_idx]
        elif day == days[next_valid_day_idx]:
            interpolate_lars[i] = lars[next_valid_day_idx]
            last_valid_day_idx += 1
            next_valid_day_idx += 1
        else:
            next_valid_day_lars = lars[next_valid_day_idx]
            last_valid_day_lars = lars[last_valid_day_idx]
            interpolate_lars[i] = \
                (next_valid_day_lars - last_valid_day_lars) / (days[next_valid_day_idx] - days[last_valid_day_idx]) \
                * (day - days[last_valid_day_idx]) \
                + last_valid_day_lars
    return interpolate_lars


def test_linear_interpolation():
    days = np.array([-5,-2, 0, 2, 5])

    lars = np.array([[-5, -2.5], [-2, -1], [0, 0], [1, 2], [2.5, 5]])

    interpolated_lars = linear_interpolate(lars, days)

    correct_interpolated_lars = np.array([[-5, -2.5], [-4, -2], [-3, -1.5], [-2, -1], [-1, -0.5], [0, 0],
                                          [0.5, 1], [1, 2], [1.5, 3], [2, 4], [2.5, 5]])

    assert np.all(correct_interpolated_lars == interpolated_lars), \
        "interpolated_lars = {} \n the correct lars = {}".format(interpolated_lars, correct_interpolated_lars)


def test_interpolate_data(interpolation_type, plot_interpolation=False,
                          plot_dir=None, plot_ground_truth=False):
    from src.utils.data_loader import load_data
    from src.utils.available_data import DATA_DIR_DICT
    import git
    import os

    data_type = "count"
    data_dir = DATA_DIR_DICT[data_type]
    repo = git.Repo('.', search_parent_directories=True)
    repo_dir = repo.working_tree_dir  # microbialdynamics
    data_dir = os.path.join(repo_dir, data_dir)
    Dx = 10
    training_sample_idx = None
    test_sample_idx = None

    hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, \
    extra_inputs_train, extra_inputs_test = \
        load_data(data_dir, Dx, False,
                  training_sample_idx=training_sample_idx, test_sample_idx=test_sample_idx)

    if interpolation_type == 'count_clv':
        from src.utils.available_data import INTERPOLATION_DATA_DICT
        import pickle

        interpolation_data_dir = INTERPOLATION_DATA_DICT['count_clv']
        interpolation_data_dir = os.path.join(repo_dir, interpolation_data_dir)
        with open(interpolation_data_dir, "rb") as f:
            interpolation_data = pickle.load(f)
    else:
        interpolation_data = None

    hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, \
    _mask_train, _mask_test, time_interval_train, time_interval_test, extra_inputs_train, extra_inputs_test = \
        interpolate_data(hidden_train, hidden_test, obs_train, obs_test, input_train, input_test,
                         extra_inputs_train, extra_inputs_test,
                         interpolation_type=interpolation_type, interpolation_data=interpolation_data)

    if plot_interpolation:
        from src.rslts_saving.rslts_saving import plot_obs_bar_plot
        plot_dir = os.path.join(repo_dir, plot_dir)

        plot_obs_bar_plot(obs_train, batch_mask=None, rslt_dir=plot_dir+"/train")

    if plot_ground_truth:
        from src.rslts_saving.rslts_saving import plot_obs_bar_plot
        plot_dir = os.path.join(repo_dir, plot_dir)

        plot_obs_bar_plot(obs_train, batch_mask=_mask_train, rslt_dir=plot_dir + "/train")


def trainer_interpolation_helper(data, y_hat_vals, masks):
    """

    :param data: a list, each is an ndarray of shape (time, Dy)
    :param y_hat_vals: a list, each is an ndarry of shape (time, Dy)
    :param masks: a list, each is an ndarray of shape (time,)
    :return: updated_data: a list, each is an ndarray of shape (time, ), updated_extra_inputs: a list, each is an ndarray of shape (time, )
    """

    print("starting inteprolation")
    def interpolate(y, y_hat_val, mask):
        pseudo_count = 1
        updated_y = np.round(y_hat_val[0]).astype(int) + pseudo_count
        updated_y[mask] = y[mask]
        return updated_y

    updated_data = list(map(interpolate, data, y_hat_vals, masks))
    updated_extra_inputs = [np.sum(updated_y, axis=-1) for updated_y in updated_data]   

    return updated_data, updated_extra_inputs


if __name__ == "__main__":
    # test_linear_interpolation()
    print("hey")

    test_interpolate_data(interpolation_type="linear_lar", plot_interpolation=False,
                    plot_dir='test_interpolation/linear_lar')

    test_interpolate_data(interpolation_type="gp_lar", plot_interpolation=False, plot_dir='test_interpolation/gp_lar')



    # test_interpolate_data(interpolation_type="gp", plot_interpolation=False, plot_dir="test_interpolation/gp")


    test_interpolate_data(interpolation_type="count_clv",
                          plot_interpolation=True, plot_dir="test_interpolation/clv")


    test_interpolate_data(interpolation_type=None, plot_interpolation=False,
                          plot_dir="test_interpolation/ground_truth", plot_ground_truth=False)






