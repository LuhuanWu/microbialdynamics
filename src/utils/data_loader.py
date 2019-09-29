import pickle
import numpy as np


def load_data(path, Dx, isPython2, training_sample_idx=None, test_sample_idx=None):
    with open(path, "rb") as handle:
        if isPython2:
            data = pickle.load(handle, encoding="latin1")
        else:
            data = pickle.load(handle)

    obs_train = data["Ytrain"]
    obs_test = data["Ytest"]
    input_train = data["Vtrain"]
    input_test = data["Vtest"]

    if "Xtrain" in data and "Xtest" in data:
        hidden_train = data["Xtrain"]
        hidden_test = data["Xtest"]
    else:
        hidden_train = [np.zeros((obs.shape[0], Dx)) for obs in obs_train]
        hidden_test = [np.zeros((obs.shape[0], Dx)) for obs in obs_test]

    if "counts_train" in data and "counts_test" in data:
        extra_inputs_train = data["counts_train"]
        extra_inputs_test = data["counts_test"]
    else:
        extra_inputs_train = [None for _ in range(len(obs_train))]
        extra_inputs_test = [None for _ in range(len(obs_test))]

    obs = obs_train + obs_test
    hidden = hidden_train + hidden_test
    input = input_train + input_test
    extra_inputs = extra_inputs_train + extra_inputs_test

    if training_sample_idx is not None:
        assert isinstance(training_sample_idx, list)
        # use selected samples for training

        hidden_train = [np.array(hidden[i]) for i in training_sample_idx]
        obs_train = [np.array(obs[i]) for i in training_sample_idx]
        input_train = [np.array(input[i]) for i in training_sample_idx]
        extra_inputs_train = [np.array(extra_inputs[i]) for i in training_sample_idx]

    if test_sample_idx is not None:
        assert isinstance(test_sample_idx, list)
        # use selected samples for test

        hidden_test = [np.array(hidden[i]) for i in test_sample_idx]
        obs_test = [np.array(obs[i]) for i in test_sample_idx]
        input_test = [np.array(input[i]) for i in test_sample_idx]
        extra_inputs_test = [np.array(extra_inputs[i]) for i in test_sample_idx]

    return hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, extra_inputs_train, extra_inputs_test

