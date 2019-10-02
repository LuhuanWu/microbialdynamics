# generate plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import git
import pickle
import click

from src.utils.available_data import DATA_DIR_DICT


@click.command()
@click.option('--data_type', default="clv_count_Dx_10_obs_10s_no_noise")
@click.option('--viz_dir', default='visualizations', help='directory of visualization plots')
@click.option('--plot_idx', default=','.join(map(str, np.arange(10))), help="idx of the samples to plot")
def generate_visualizations(data_type, viz_dir, plot_idx):
    if data_type is None:
        print("Must provide data_type.")

    repo = git.Repo('.', search_parent_directories=True)
    repo_dir = repo.working_tree_dir  # microbialdynamics

    data_dir = DATA_DIR_DICT[data_type]
    data_dir = os.path.join(repo_dir, data_dir)

    print(data_dir)

    with open(data_dir, "rb") as f:
        data = pickle.load(f)

    obs_train = data['Ytrain']
    obs_test = data['Ytest']
    inputs_train = data['Vtrain']
    inputs_test = data['Vtest']

    obs = obs_train + obs_test
    inputs = inputs_train + inputs_test

    if plot_idx == '-1':
        plot_idx = range(len(obs))
    else:
        print(plot_idx)
        plot_idx = [int(i) for i in plot_idx.split(",")]

        obs = [obs[i] for i in plot_idx]
        inputs = [inputs[i] for i in plot_idx]

    def interpolate_datapoint(obs, input):
        """

        :param obs: (n_obs, Dy + 1), [:, 0] records t of all obs
        :return:
        obs: (time, Dy)
        mask: (time, )
        time_interval: (time, )
        """
        days = obs[:, 0].astype(int)
        time = days[-1] - days[0] + 1

        mask = np.ones((time,), dtype=bool)
        time_interval = np.zeros((time,))

        i = 0
        for t in np.arange(days[0], days[-1] + 1):
            idx = t - days[0]
            if t == days[i]:
                i = i + 1
                time_interval[idx] = 0
            else:
                mask[t - days[0]] = False
                time_interval[idx] = time_interval[idx - 1] + 1

        # obs
        Dy = obs.shape[1] - 1

        interpolated_obs = np.zeros((time, Dy))
        last_valid_value = np.ones(Dy) / Dy
        i = 0
        for t in np.arange(days[0], days[-1] + 1):
            if t == days[i]:
                smoothed_obs = obs[i, 1:]
                interpolated_obs[t - days[0]] = smoothed_obs
                last_valid_value = smoothed_obs
                i += 1
            else:
                interpolated_obs[t - days[0]] = last_valid_value

        # inputs
        Dv = input.shape[1] - 1
        interpolated_input = np.zeros((time, Dv))
        for day_input in input:
            day = int(day_input[0])
            if days[0] <= day <= days[-1]:
                interpolated_input[day - days[0]] = day_input[1:]

        return interpolated_obs, interpolated_input, mask, time_interval

    def interpolate_data(batch_obs, batch_inputs):

        interpolated_batch_obs = []
        interpolated_batch_inputs = []
        batch_mask = []
        batch_time_interval = []

        for obs, inputs in zip(batch_obs, batch_inputs):
            obs, inputs, mask, time_interval = interpolate_datapoint(obs, inputs)

            interpolated_batch_obs.append(obs)
            interpolated_batch_inputs.append(inputs)
            batch_mask.append(mask)
            batch_time_interval.append(time_interval)

        return interpolated_batch_obs, interpolated_batch_inputs, batch_mask, batch_time_interval

    new_obs, new_inputs, masks, time_interval = interpolate_data(obs, inputs)

    def bar_plot(ax, obs, mask, to_normalize=True, ):
        if to_normalize:
            obs = obs / np.sum(obs, axis=-1, keepdims=True)

        time, Dy = obs.shape

        # make missing obs = 0
        masked_obs = np.zeros_like(obs)
        masked_obs[mask] = obs[mask]

        ax.set_xlabel("Time")
        bottom = np.zeros(time)
        for j in range(Dy):
            ax.bar(np.arange(time), masked_obs[:, j], bottom=bottom, edgecolor='white')
            bottom += masked_obs[:, j]

        ax.set_xticks(np.arange(time))
        sns.despine()

    def input_plot(ax, inputs):
        time, Dv = inputs.shape

        for j in range(Dv):
            has_inputs = inputs[:, j] == 1
            idx = np.arange(time)[has_inputs]
            ax.bar(idx, [1 for _ in idx], bottom=[j for _ in idx], color='blue')

        ax.set_xticks(np.arange(time))
        ax.set_yticks(np.arange(Dv))
        sns.despine()

    def plot_inputs_and_obs(obs, input, mask, to_normalize=True):
        plt.figure(figsize=(15, 10))

        ax1 = plt.subplot(2, 1, 1)
        input_plot(ax1, input)
        ax1.grid()

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        bar_plot(ax2, obs, mask, to_normalize=to_normalize)
        ax2.grid()

    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    print("Saving plots to", viz_dir)

    for i in range(len(new_obs)):
        plot_inputs_and_obs(new_obs[i], new_inputs[i], masks[i])
        plt.savefig(viz_dir + "/sample_{}".format(plot_idx[i]))
        plt.close()


if __name__ == '__main__':
    generate_visualizations()


