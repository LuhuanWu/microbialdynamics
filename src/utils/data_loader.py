import pickle
import numpy as np


def load_data(path, train_num=-1, test_num=-1):
    with open(path, "rb") as handle:
        data = pickle.load(handle)

    obs_train = data["Ytrain"]
    obs_test = data["Ytest"]
    input_train = data["Vtrain"]
    input_test = data["Vtest"]

    theta = None
    if "theta" in data:
        theta = data["theta"]

    if "Xtrain" in data and "Xtest" in data:
        hidden_train = data["Xtrain"]
        hidden_test = data["Xtest"]
    else:
        hidden_train = [None for _ in obs_train]
        hidden_test = [None for _ in obs_test]

    if "depth_train" in data and "depth_test" in data:
        depth_train = data["depth_train"]
        depth_test = data["depth_test"]
    else:
        depth_train = [obs[:, 1:].sum(axis=-1) for obs in obs_train]
        depth_test = [obs[:, 1:].sum(axis=-1) for obs in obs_test]

    params = None
    if "b" in data and "g" in data and "A" in data and "W" in data:
        b, g, A, W = data["b"], data["g"], data["A"], data["W"]
        params = (b, g, A, W)

    if train_num > 0:
        obs_train = obs_train[:train_num]
        input_train = input_train[:train_num]
        hidden_train = hidden_train[:train_num]
        depth_train = depth_train[:train_num]

    if test_num > 0:
        obs_test = obs_test[:test_num]
        input_test = input_test[:test_num]
        hidden_test = hidden_test[:test_num]
        depth_test = depth_test[:test_num]

    return hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, depth_train, depth_test, \
           theta, params
