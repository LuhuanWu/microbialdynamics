import pickle
import git
import numpy as np
from scipy.special import logsumexp

import matplotlib.pyplot as plt
import seaborn as sns

import scipy


# generate binary input vector
def simulate_single_input(time, Dv):
    """

    :param time: a scalar
    :param Dv: a scalar
    :return: input: (time, Dv)
    """
    inputs = np.zeros((time, Dv))

    number_of_non_zero_dimenions = np.random.choice(np.arange(3, 8), 1)[0]

    non_zero_dimensions = np.random.choice(Dv - 1, number_of_non_zero_dimenions, replace=False)

    for k in non_zero_dimensions:
        start = np.random.choice(time - 1, 1)[0]
        end = start + 1 + np.random.choice(np.arange(40, 80), 1)[0]
        end = min(end, time)
        inputs[start:end, k] = np.ones(end - start)

    # surgery is at the last axis
    start = np.random.randint(int(0.3 * time), int(0.6 * time))
    end = time
    inputs[start:end, -1] = np.ones(end - start)

    return inputs


def simulate_clv_with_inputs(A, g, Wg, W1, W2, f_cov, N, inputs):
    """
    x_t + g_t + Wg v_t + (A+ A(v_t)) * p_t where A(v_t) = (W1 * vt) * W2
    :param A: (Dx, Dx + 1)
    :param g: (Dx, )
    :param Wg: (Dx, Dv)
    :param W1: (Dx, Dv)
    :param W2: (1, Dx+1)
    :param f_cov:
    :param N:
    :param inputs: (time, Dv)
    :return:
    """
    latent_dim = A.shape[0]  # Dx
    ndays, input_dim = inputs.shape
    x = []
    y_count = []
    y_percentage = []

    # modify the mu
    mu = np.random.multivariate_normal(mean=np.zeros(latent_dim), cov=np.eye(latent_dim))
    for t in range(ndays):
        xt = mu

        # increase dimension by 1
        xt1 = np.concatenate((xt, np.array([0])))
        pt = np.exp(xt1 - logsumexp(xt1))

        # simulate total number of reads with over-dispersion
        logN = np.random.normal(loc=np.log(N), scale=0.5)
        Nt = np.random.poisson(np.exp(logN))

        yt_count = np.random.multinomial(Nt, pt).astype(float)
        yt_percentage = yt_count / np.sum(yt_count)

        x.append(xt)
        y_count.append(yt_count)
        y_percentage.append(yt_percentage)

        transition_noise = np.random.multivariate_normal(mean=np.zeros(latent_dim), cov=np.diag(f_cov))
        vt = inputs[t]

        # (Dx, Dv) * (Dv, 1) --> (Dx, 1)
        Aofv = np.matmul(W1, vt[:, None])
        # (Dx, 1) * (1, Dx + 1) --> (Dx, Dx+1)
        Aofv = np.matmul(Aofv, W2)
        mu = xt + g + Wg.dot(vt) + (A + Aofv).dot(pt) + transition_noise
        mu = np.clip(mu, -3, 3)

    return np.array(x), np.array(y_count), np.array(y_percentage)


def get_data_to_pickle(A, g, Wa, W1, W2, f_cov, N, time_min, time_max, scale):
    # create data with missing observation
    simulation_time = time_max * scale

    x_train = []
    x_test = []
    y_count_train = []
    y_count_test = []
    y_percentage_train = []
    y_percentage_test = []
    v_train = []
    v_test = []

    # create inputs
    batch_inputs = [simulate_single_input(simulation_time, ninput) for _ in range(n_train + n_test)]

    for i in range(n_train + n_test):
        v = batch_inputs[i]  # (time, Dv)
        x, y_count, y_percentage = simulate_clv_with_inputs(A, g, Wg, W1, W2, f_cov, N, v)

        idx = np.arange(time_max) * scale
        ndays = np.random.randint(time_min, time_max)
        start = np.random.randint(time_max - ndays)
        idx = idx[start:start + ndays]

        x = x[idx]
        y_count = y_count[idx]
        y_percentage = y_percentage[idx]
        v = v[idx]

        # make missing observations, the first day cannot be missing
        obs_percentage = np.random.choice([0.4, 0.5, 0.6, 0.7, 0.8], p=[0.1, 0.2, 0.2, 0.2, 0.3])
        # obs_percentage = 0.999
        obsed_days = np.random.choice(np.arange(1, ndays), int(ndays * obs_percentage), replace=False)
        obsed_days = np.sort(np.concatenate(([0], obsed_days)))

        y_percentage = y_percentage[obsed_days]
        x = x[obsed_days]
        y_count = y_count[obsed_days]

        days = np.arange(ndays)[:, np.newaxis]
        y_count = np.concatenate([days[obsed_days], y_count], axis=-1)
        y_percentage = np.concatenate([days[obsed_days], y_percentage], axis=-1)
        v = np.concatenate([days, v], axis=-1)

        if i < n_train:
            x_train.append(x)
            y_count_train.append(y_count)
            y_percentage_train.append(y_percentage)
            v_train.append(v)
        else:
            x_test.append(x)
            y_count_test.append(y_count)
            y_percentage_test.append(y_percentage)
            v_test.append(v)

    counts_train = []
    for single_obs in y_count_train:
        single_counts = single_obs[:, 1:].sum(axis=-1)
        counts_train.append(single_counts)

    counts_test = []
    for single_obs in y_count_test:
        single_counts = single_obs[:, 1:].sum(axis=-1)
        counts_test.append(single_counts)

    p_data = {}
    p_data["Xtrain"] = x_train
    p_data["Xtest"] = x_test
    p_data["Ytrain"] = y_percentage_train
    p_data["Ytest"] = y_percentage_test
    p_data["Vtrain"] = v_train
    p_data["Vtest"] = v_test

    p_data["A"] = A
    p_data["Wa"] = Wa
    p_data["g"] = g
    p_data["Wg"] = Wg
    p_data["f_cov"] = f_cov
    p_data["N"] = N

    c_data = {}
    c_data["Xtrain"] = x_train
    c_data["Xtest"] = x_test
    c_data["Ytrain"] = y_count_train
    c_data["Ytest"] = y_count_test
    c_data["Vtrain"] = v_train
    c_data["Vtest"] = v_test
    c_data["counts_train"] = counts_train
    c_data["counts_test"] = counts_test

    c_data["A"] = A
    c_data["Wa"] = Wa
    c_data["g"] = g
    c_data["Wg"] = Wg
    c_data["f_cov"] = f_cov
    c_data["N"] = N

    return p_data, c_data


for Dx in range(1, 11):
    for scale in [1, 4]:
        print("Dx = {}, scale = {}".format(Dx, scale))
        ntaxa = Dy = Dx + 1
        ninput = Dv = 10  # including surgery
        n_train, n_test = 200, 30
        time_min = 30
        time_max = 50

        A = np.random.normal(loc=0,    scale=0.05, size=(Dx, Dx + 1))
        g = np.random.normal(loc=0,    scale=0.05, size=(Dx,))
        Wg = np.random.normal(loc=-0.2, scale=0.2, size=(Dx, Dv))
        W1 = np.random.normal(loc=-0.2, scale=0.2, size=(Dx, Dv))
        W2 = np.random.normal(loc=0.2, scale=0.2, size=(1, Dx + 1))

        f_cov = np.random.uniform(0, 1, ntaxa - 1)
        N = 10000  # sequencing reads parameter

        p_data, c_data = get_data_to_pickle(A, g, Wg, W1, W2, f_cov, N, time_min, time_max, scale)

        repo = git.Repo('.', search_parent_directories=True)
        repo_dir = repo.working_tree_dir  # microbialdynamics

        # percentage
        data_dir = repo_dir + "/data/clv/data/clv_percentage_Dx_{}_scale_{}.p".format(Dx, scale)
        with open(data_dir, "wb") as f:
            pickle.dump(p_data, f)

        # count
        data_dir = repo_dir + "/data/clv/data/clv_count_Dx_{}_scale_{}.p".format(Dx, scale)
        with open(data_dir, "wb") as f:
            pickle.dump(c_data, f)


