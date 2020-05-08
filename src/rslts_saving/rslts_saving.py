import numpy as np

import os
import json
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import logsumexp, softmax

from src.rslts_saving.datetools import addDateTime


def create_RLT_DIR(Experiment_params):
    # create the dir to save data
    # Experiment_params is a dict containing param_name&param pair
    # Experiment_params must contain "rslt_dir_name":rslt_dir_name
    cur_date = addDateTime()

    local_rlt_root = "rslts/" + Experiment_params["rslt_dir_name"] + "/"

    params_str = ""
    for param_name, param in Experiment_params.items():
        if param_name == "rslt_dir_name":
            continue
        params_str += "_" + param_name + "_" + str(param)

    RLT_DIR = os.getcwd().replace("\\", "/") + "/" + local_rlt_root + cur_date[1:] + params_str + "/"

    if not os.path.exists(RLT_DIR):
        os.makedirs(RLT_DIR)

    return RLT_DIR


def save_experiment_param(RLT_DIR, FLAGS):
    params_dict = {}
    params_list = sorted([param for param in dir(FLAGS) if param
                          not in ['h', 'help', 'helpfull', 'helpshort']])

    print("Experiment_params:")
    for param in params_list:
        params_dict[param] = str(getattr(FLAGS, param))
        print("\t" + param + ": " + str(getattr(FLAGS, param)))

    with open(RLT_DIR + "param.json", "w") as f:
        json.dump(params_dict, f, indent=4, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    # Special json encoder for numpy types
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                      np.int16, np.int32, np.int64, np.uint8,
                      np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                        np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def plot_training_data(RLT_DIR, hidden_train, obs_train, saving_num=20):
    # Plot and save training data
    if not os.path.exists(RLT_DIR + "Training Data"):
        os.makedirs(RLT_DIR + "Training Data")
    saving_num = min(len(hidden_train), saving_num)
    for i in range(saving_num):
        plt.figure()
        plt.title("Training Time Series")
        plt.xlabel("Time")
        plt.plot(hidden_train[i], c="red")
        plt.plot(obs_train[i], c="blue")
        sns.despine()
        plt.savefig(RLT_DIR + "Training Data/{}".format(i))
        plt.close()


def plot_learning_results(RLT_DIR, Xs_val, hidden_train, saving_num=20):
    # Plot and save learning results
    if not os.path.exists(RLT_DIR + "Learning Results"):
        os.makedirs(RLT_DIR + "Learning Results")
    saving_num = min(len(hidden_train), saving_num)
    for i in range(saving_num):
        for j in range(Xs_val.shape[-1]):
            plt.figure()
            plt.title("hidden state {}".format(j))
            plt.xlabel("Time")
            plt.plot(np.mean(Xs_val[i, :, :, j], axis=1), alpha=0.5, c="black")
            plt.plot(hidden_train[i, :, j], c="yellow")
            plt.legend(["prediction", "ground truth"])
            sns.despine()
            plt.savefig(RLT_DIR + "Learning Results/h_dim_{}_idx_{}".format(j, i))
            plt.close()


def plot_log_ZSMC(RLT_DIR, log_ZSMC_trains, log_ZSMC_tests, start_idx, print_freq):
    epoch = start_idx + np.arange(len(log_ZSMC_trains)) * print_freq
    plt.figure()
    plt.plot(epoch, log_ZSMC_trains)
    plt.plot(epoch, log_ZSMC_tests)
    plt.legend(["log_ZSMC_train", "log_ZSMC_test"])
    sns.despine()
    plt.savefig(RLT_DIR + "log_ZSMC")
    plt.show()


def plot_MSEs(RLT_DIR, MSE_trains, MSE_tests, print_freq):
    if not os.path.exists(RLT_DIR + "MSE"):
        os.makedirs(RLT_DIR + "MSE")
    # Plot and save losses
    plt.figure()
    for i in range(len(MSE_trains)):
        plt.plot(MSE_trains[i])
        plt.plot(MSE_tests[i])
        plt.xlabel("k")
        plt.legend(["MSE_train", "MSE_test"])
        sns.despine()
        plt.savefig(RLT_DIR + "MSE/epoch_{}".format(i * print_freq))
        plt.close()


def plot_R_square(RLT_DIR, R_square_trains, R_square_tests, start_idx, print_freq):
    if not os.path.exists(RLT_DIR + "R_square"):
        os.makedirs(RLT_DIR + "R_square")
    # Plot and save losses
    plt.figure()
    for i in range(len(R_square_trains)):
        plt.plot(R_square_trains[i])
        plt.plot(R_square_tests[i])
        plt.ylim([0.0, 1.05])
        plt.xlabel("K")
        plt.legend(["Train $R^2_k$", "Test $R^2_k$"], loc='best')
        sns.despine()
        plt.savefig(RLT_DIR + "R_square/epoch_{}".format(start_idx + i * print_freq))
        plt.close()


def plot_R_square_epoch(RLT_DIR, R_square_trains, R_square_tests, epoch):
    if not os.path.exists(RLT_DIR + "R_square"):
        os.makedirs(RLT_DIR + "R_square")
    plt.figure()
    plt.plot(R_square_trains)
    plt.plot(R_square_tests)
    plt.ylim([0.0, 1.05])
    plt.xlabel("K")
    plt.legend(["Train $R^2_k$", "Test $R^2_k$"], loc='best')
    sns.despine()
    plt.savefig(RLT_DIR + "R_square/epoch_{}".format(epoch))
    plt.close()


def plot_fhn_results(RLT_DIR, Xs_val):
    if not os.path.exists(RLT_DIR + "FHN 2D plots"):
        os.makedirs(RLT_DIR + "FHN 2D plots")
    for i in range(Xs_val.shape[0]):
        plt.figure()
        plt.title("hidden state for all particles")
        plt.xlabel("x_dim 1")
        plt.ylabel("x_dim 2")
        for j in range(Xs_val.shape[2]):
            plt.plot(Xs_val[i, :, j, 0], Xs_val[i, :, j, 1])
        sns.despine()
        plt.savefig(RLT_DIR + "/FHN 2D plots/All_x_paths_{}".format(i))
        plt.close()


def plot_lorenz_results(RLT_DIR, Xs_val):
    if not os.path.exists(RLT_DIR + "Lorenz 3D plots"):
        os.makedirs(RLT_DIR + "Lorenz 3D plots")
    for i in range(Xs_val.shape[0]):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        plt.title("hidden state for all particles")
        ax.set_xlabel("x_dim 1")
        ax.set_ylabel("x_dim 2")
        ax.set_zlabel("x_dim 3")
        for j in range(Xs_val.shape[2]):
            ax.plot(Xs_val[i, :, j, 0], Xs_val[i, :, j, 1], Xs_val[i, :, j, 2])
        plt.savefig(RLT_DIR + "/Lorenz 3D plots/All_x_paths_{}".format(i))
        plt.close()



def plot_y_hat(RLT_DIR, ys_hat_val, obs, mask, saving_num=20):
    # yhat, a list of length K+1, each item is a list of k-step prediction for #n_test, each of which is an array
    # (T-k, Dy)
    # obs, a list of length #n_test, each item is an ndarray of shape (time, Dy)
    # mask: a list of masks, each item is an array of shape (time, )

    if saving_num == 0:
        return

    if not os.path.exists(RLT_DIR):
        os.makedirs(RLT_DIR)

    Dy = obs[0].shape[1]
    saving_num = min(len(obs), saving_num)

    for i in range(saving_num):
        for j in range(Dy):
            plt.figure()
            plt.title("obs dim {}".format(j))
            plt.xlabel("Time")

            time = obs[i].shape[0]

            masked_time = np.arange(time)[mask[i]]
            masked_obs = obs[i][mask[i]]

            plt.plot(masked_time, masked_obs[:,j], '-o')

            for k, ys_k_hat_val in enumerate(ys_hat_val):
                masked_time = np.arange(k, time)[mask[i][k:]]
                #masked_yhat = ys_k_hat_val[i][mask[i][k:]]
                masked_yhat = ys_k_hat_val[i]
                plt.plot(masked_time, masked_yhat[:,j], "--", label="k={}".format(k))

            plt.legend()
            sns.despine()
            plt.savefig(RLT_DIR + "/obs_dim_{}_idx_{}".format(j, i))
            plt.close()


def plot_obs_bar_plot(batch_obs_original, batch_mask=None, to_normalize=True, rslt_dir="obs_bar_plots"):

    if to_normalize:
        batch_obs = []
        for i, obs in enumerate(batch_obs_original):
            batch_obs.append(obs / np.sum(obs, axis=-1, keepdims=True))
    else:
        batch_obs = batch_obs_original

    Dy = batch_obs[0].shape[-1]

    for i, obs in enumerate(batch_obs):
        time = obs.shape[0]
        if batch_mask is None:
            masked_obs = obs
        else:
            masked_obs = np.zeros_like(obs)
            masked_obs[batch_mask[i]] = obs[batch_mask[i]]

        plt.figure(figsize=(15,5))
        plt.title("obs idx {} ground truth".format(i))
        plt.xlabel("Time")
        bottom = np.zeros(masked_obs.shape[0])
        for j in range(Dy):
            plt.bar(np.arange(time), masked_obs[:, j], bottom=bottom, edgecolor='white')
            bottom += masked_obs[:, j]

        plt.xticks(np.arange(time))
        sns.despine()
        if not os.path.exists(rslt_dir):
            os.makedirs(rslt_dir)
        plt.savefig(rslt_dir + "/obs_idx_{} truth".format(i))
        plt.close()


def plot_y_hat_bar_plot(RLT_DIR, ys_hat_val, original_obs, mask, saving_num=20, to_normalize=True):
    # ys_hat_val, a list, a list of length K+1, each item is a list of k-step prediction for #n_test,
    # each of which is an array (T-k, Dy)
    if saving_num == 0:
        return
    obs = [np.array(obs_i) for obs_i in original_obs]
    if to_normalize:
        for i, obs_i in enumerate(obs):
            obs[i] = obs_i / np.sum(obs_i, axis=-1, keepdims=True)

        for k, ys_k_hat_val in enumerate(ys_hat_val):
            for i, ys_k_hat_val_i in enumerate(ys_k_hat_val):
                ys_k_hat_val[i] = ys_k_hat_val_i / np.sum(ys_k_hat_val_i, axis=-1, keepdims=True)

    if not os.path.exists(RLT_DIR):
        os.makedirs(RLT_DIR)

    Dy = obs[0].shape[1]
    saving_num = min(len(obs), saving_num)

    for i in range(saving_num):
        time = obs[i].shape[0]

        masked_obs = np.zeros_like(obs[i])
        masked_obs[mask[i]] = obs[i][mask[i]]

        plt.figure(figsize=(15,5))
        plt.title("obs idx {} ground truth".format(i))
        plt.xlabel("Time")
        bottom = np.zeros(masked_obs.shape[0])
        for j in range(Dy):
            plt.bar(np.arange(time), masked_obs[:, j], bottom=bottom, edgecolor='white')
            bottom += masked_obs[:, j]

        plt.xticks(np.arange(time))
        sns.despine()
        plt.savefig(RLT_DIR + "/obs_idx_{} truth".format(i))
        plt.close()

        for k, ys_k_hat_val in enumerate(ys_hat_val):
            masked_yhat = np.zeros((time, Dy))  # (full_ndays, Dy)

            masked_yhat[k:][mask[i][k:]] = ys_k_hat_val[i]

            plt.figure(figsize=(15,5))
            plt.title("obs idx {} {}-step prediction".format(i, k))
            plt.xlabel("Time")
            bottom = np.zeros(masked_yhat.shape[0])
            for j in range(Dy):
                plt.bar(np.arange(0, time), masked_yhat[:, j], bottom=bottom, edgecolor='white')
                bottom += masked_yhat[:, j]

            plt.xticks(np.arange(0, time))
            sns.despine()
            plt.savefig(RLT_DIR + "/obs_idx_{}_{}_step".format(i, k))
            plt.close()


def plot_x_bar_plot(RLT_DIR, xs_val, saving_num=20):
    # ys_hat_val, a list, a list of length K+1, each item is a list of k-step prediction for #n_test,
    # each of which is an array (T-k, Dy)
    if not os.path.exists(RLT_DIR):
        os.makedirs(RLT_DIR)

    if saving_num == 0:
        return
    xs_val = [np.mean(x_traj, axis=1) for x_traj in xs_val]

    saving_num = min(len(xs_val), saving_num)

    for i in range(saving_num):
        x_traj = xs_val[i]
        time = x_traj.shape[0]
        percentage = softmax(x_traj, axis=-1)

        plt.figure(figsize=(15, 5))
        plt.title("topic proportion idx {}".format(i))
        plt.xlabel("Time")
        bottom = np.zeros(time)
        for j in range(percentage.shape[1]):
            plt.bar(np.arange(time), percentage[:, j], bottom=bottom, edgecolor='white')
            bottom += percentage[:, j]

        plt.xticks(np.arange(0, time, time // 15))
        sns.despine()
        plt.savefig(RLT_DIR + "/topic_proportion_idx_{}".format(i))
        plt.close()


def plot_topic_bar_plot(RLT_DIR, beta, epoch=None):
    if not os.path.exists(RLT_DIR):
        os.makedirs(RLT_DIR)

    plt.figure(figsize=(15, 5))
    plt.title("topic content")
    plt.xlabel("topic")
    plt.ylabel("taxon")
    n_topics = beta.shape[0]
    bottom = np.zeros(n_topics)
    for j in range(beta.shape[1]):
        plt.bar(np.arange(n_topics), beta[:, j], bottom=bottom, edgecolor='white')
        bottom += beta[:, j]
    plt.xticks(np.arange(n_topics))
    sns.despine()
    if epoch is None:
        plt.savefig(RLT_DIR + "/topic_content")
    else:
        plt.savefig(RLT_DIR + "/topic_content_epoch{}".format(epoch))
    plt.close()


def plot_topic_bar_plot_across_time(RLT_DIR, beta_logs, saving_num=20):

    if not os.path.exists(RLT_DIR):
        os.makedirs(RLT_DIR)

    saving_num = min(len(beta_logs), saving_num)

    for i in range(saving_num):
        beta_log = beta_logs[i]
        beta_log = np.mean(beta_log, axis=1)  # (time, Dx+1, Dy-1)
        beta = softmax(beta_log, axis=-1)  # (time, Dx+1, Dy)

        time, n_topics, n_taxa = beta.shape

        for j in range(n_topics):
            plt.figure(figsize=(15, 5))
            plt.title("taxon proportion idx {} topic {}".format(i, j))
            plt.xlabel("time")
            bottom = np.zeros(time)
            for k in range(n_taxa):
                plt.bar(np.arange(time), beta[:, j, k], bottom=bottom, edgecolor='white')
                bottom += beta[:, j, k]

            plt.xticks(np.arange(0, time, time // 15))
            plt.tight_layout()
            sns.despine()

            plt.savefig(RLT_DIR + "/traj_{}_group_{}".format(i, j))
            plt.close()


def plot_topic_bar_plot_while_training(ax, beta, epoch):
    # ys_hat_val, a list, a list of length K+1, each item is a list of k-step prediction for #n_test,
    # each of which is an array (T-k, Dy)
    
    ax.clear()
    n_topics = beta.shape[0]
    bottom = np.zeros(n_topics)
    for j in range(beta.shape[1]):
        ax.bar(np.arange(n_topics), beta[:, j], bottom=bottom, edgecolor='white')
        bottom += beta[:, j]
    sns.despine()
    ax.set_title("topic content in epoch {}".format(epoch))


def plot_topic_taxa_matrix_while_training(ax, beta, epoch, cbar_ax):
    # beta shape is (n_topics, n_taxa)
    ax.clear()
    
    sns.heatmap(beta, ax=ax, vmin=0, vmax=1, cmap="BuGn", square=True, cbar_ax=cbar_ax)
    ax.set_xlabel('taxon')
    ax.set_ylabel('topic')
    ax.set_title("beta in epoch {}".format(epoch))


def plot_x_bar_plot_while_training(axs, xs_val):
    # ys_hat_val, a list, a list of length K+1, each item is a list of k-step prediction for #n_test,
    # each of which is an array (T-k, Dy)

    xs_val = [np.mean(x_traj, axis=1) for x_traj in xs_val]
    Dy = xs_val[0].shape[1] + 1

    for i, ax in enumerate(axs):
        ax.clear()
        
        x_traj = xs_val[i]
        time = x_traj.shape[0]
        percentage = np.concatenate((x_traj, np.zeros((time, 1))), axis=-1)  # (n_days, Dy+1)
        percentage = np.exp(percentage - logsumexp(percentage, axis=-1, keepdims=True))

        ax.set_title("topic proportion idx {}".format(i))
        ax.set_xlabel("Time")
        bottom = np.zeros(time)
        for j in range(Dy):
            ax.bar(np.arange(time), percentage[:, j], bottom=bottom, edgecolor='white')
            bottom += percentage[:, j]

        ax.set_xticks(np.arange(time))
        sns.despine()


def plot_interaction_matrix(RLT_DIR, inferred, truth):
    A = inferred["A"]
    A_beta = inferred["A_beta"]
    theta = inferred["theta"]
    g_beta = inferred["g_beta"]

    A_beta = np.clip(A_beta, -2, 2)

    # interaction
    Dx = A.shape[0]
    Dy = A_beta.shape[1]
    if not os.path.exists(RLT_DIR):
        os.makedirs(RLT_DIR)
    sns.heatmap(A, cmap="seismic", center=0, square=True, linewidth=0.5)
    plt.plot([0, Dx], [0, Dx], "k")
    plt.tight_layout()
    plt.savefig(RLT_DIR + "/A")
    plt.close()

    for i, A_group in enumerate(A_beta):
        sns.heatmap(A_group, cmap="seismic", center=0, square=True, linewidth=0.5)
        plt.plot([0, Dy], [0, Dy], "k")
        plt.tight_layout()
        plt.savefig(RLT_DIR + "/A_beta_{}".format(i))
        plt.close()

    if "A" in truth:
        A_truth = truth["A"]
        sns.heatmap(A_truth, cmap="seismic", center=0, square=True, linewidth=0.5)
        plt.plot([0, Dx], [0, Dx], "k")
        plt.tight_layout()
        plt.savefig(RLT_DIR + "/A_truth")
        plt.close()

    if "A_g" in truth:
        A_beta_truth = truth["A_g"]
        A_beta_truth = np.clip(A_beta_truth, -2, 2)
        for i, A_group in enumerate(A_beta_truth):
            sns.heatmap(A_group, cmap="seismic", center=0, square=True, linewidth=0.5)
            plt.plot([0, Dy], [0, Dy], "k")
            plt.tight_layout()
            plt.savefig(RLT_DIR + "/A_beta_truth_{}".format(i))
            plt.close()

    # growth
    if "g_g" in truth:
        g_beta_truth = truth["g_g"]
        sns.heatmap(np.concatenate([g_beta_truth[None, :], g_beta], axis=0),
                    cmap="seismic", center=0, square=True, linewidth=0.5)
        ticks = ["truth"] + ["group {}".format(i) for i in range(Dx)]
        plt.yticks(0.5 + np.arange(Dx + 1), ticks, rotation=0)
        plt.tight_layout()
        plt.savefig(RLT_DIR + "/g")
        plt.close()

    # theta
    sns.heatmap(theta.T, cmap="seismic", center=0, square=True, linewidth=0.5)
    yticks = ["taxon {}".format(i) for i in range(Dy)]
    plt.yticks(0.5 + np.arange(Dy), yticks, rotation=0)
    xticks = ["group {}".format(i) for i in range(Dx)]
    plt.xticks(0.5 + np.arange(Dx), xticks, rotation=45)
    plt.tight_layout()
    plt.savefig(RLT_DIR + "/theta")
    plt.close()