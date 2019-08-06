import pickle
import numpy as np


def load_data(path, Dx, isPython2, q_uses_true_X):
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

    return hidden_train, hidden_test, obs_train, obs_test, input_train, input_test
