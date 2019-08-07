import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import os


def plot_lorenz_results(RLT_DIR, Xs_val):
    if not os.path.exists(RLT_DIR + "/Lorenz 3D plots"):
        os.makedirs(RLT_DIR + "/Lorenz 3D plots")
    for i, X in enumerate(Xs_val):
        X = np.mean(X, axis=1)
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        plt.title("hidden state for all particles")
        ax.set_xlabel("x_dim 1")
        ax.set_ylabel("x_dim 2")
        ax.set_zlabel("x_dim 3")
        ax.plot(X[:, 0], X[:, 1], X[:, 2])
        plt.savefig(RLT_DIR + "/Lorenz 3D plots/All_x_paths_{}".format(i))
        plt.close()
