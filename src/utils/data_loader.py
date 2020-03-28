import pickle
import numpy as np


def load_data(path, Dx, train_num=-1, test_num=-1):
    with open(path, "rb") as handle:
        data = pickle.load(handle)

    obs_train = data["Ytrain"]
    obs_test = data["Ytest"]
    input_train = data["Vtrain"]
    input_test = data["Vtest"]

    if "Xtrain" in data and "Xtest" in data:
        hidden_train = data["Xtrain"]
        hidden_test = data["Xtest"]
    else:
        hidden_train = [None for _ in obs_train]
        hidden_test = [None for _ in obs_test]

    if "counts_train" in data and "counts_test" in data:
        extra_inputs_train = data["counts_train"]
        extra_inputs_test = data["counts_test"]
    else:
        extra_inputs_train = [None for _ in range(len(obs_train))]
        extra_inputs_test = [None for _ in range(len(obs_test))]

    if train_num > 0:
        obs_train = [np.array(obs) for obs in obs_train[:train_num]]
        input_train = [np.array(input) for input in input_train[:train_num]]
        if hidden_train[0] is not None:
            hidden_train = [np.array(hidden) for hidden in hidden_train[:train_num]]
        if extra_inputs_train[0] is not None:
            extra_inputs_train = [np.array(extra_inputs) for extra_inputs in extra_inputs_train[:train_num]]

    if test_num > 0:
        obs_test = [np.array(obs) for obs in obs_test[:test_num]]
        input_test = [np.array(input) for input in input_test[:test_num]]
        if hidden_test[0] is not None:
            hidden_test = [np.array(hidden) for hidden in hidden_test[:test_num]]
        if extra_inputs_test[0] is not None:
            extra_inputs_test = [np.array(extra_inputs) for extra_inputs in extra_inputs_test[:test_num]]

    return hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, \
        extra_inputs_train, extra_inputs_test
