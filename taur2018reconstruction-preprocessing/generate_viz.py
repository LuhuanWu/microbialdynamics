# generate plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import click


@click.command()
@click.option('--csv_dir', default="taur-otu-table-filtered.csv", help='csv directory to load the data from')
@click.option('--k', default=10, help='number of top species to choose')
@click.option('--viz_dir', default='visualizations', help='directory of visualization plots')
def generate_visualizations(csv_dir, k, viz_dir):
    print("Loading data from ", csv_dir)
    f2 = pd.read_csv(csv_dir, header=None)

    # select top Ks
    data2 = f2.to_numpy()
    measure_pid = f2.loc[0:0, 1:]
    measure_pid = np.array(measure_pid, dtype=int)
    measure_pid = measure_pid.reshape((-1))
    f2pid = np.unique(measure_pid)
    print("There are {} patients".format(len(f2pid)))

    count = data2[2:-1, 1:].T.astype(int)
    print("count shape (measures, microbio speicies):", count.shape)
    count_sum_across_spieces = np.sum(count, axis=1, keepdims=True)
    percentage = count / count_sum_across_spieces

    percentage_sum_across_measures = np.sum(percentage, axis=0)
    spieces_name = data2[2:-1, 0]

    top_k_spieces_idx = np.argsort(percentage_sum_across_measures)[-k:][::-1]
    print("top_k_spieces_idx:", top_k_spieces_idx)
    for i, idx in enumerate(top_k_spieces_idx):
        print("top {:>2}, idx {:>4}, percentage_sum {:>5.2f}, name {}".format(i, idx, percentage_sum_across_measures[idx], spieces_name[idx]))

    # obs
    obs = []

    dates = data2[1, 1:].astype(int)
    percentage_obs = np.zeros((percentage.shape[0], k + 2))
    percentage_obs[:, 0] = dates
    percentage_obs[:, 1:-1] = percentage[:, top_k_spieces_idx]
    percentage_obs[:, -1] = 1 - np.sum(percentage_obs[:, 1:], axis=-1)

    pid_measure_start = 0
    for pid in f2pid:
        patient_obs = []
        for i in range(pid_measure_start, len(measure_pid)):
            if measure_pid[i] != pid:
                pid_measure_start = i
                break
            patient_obs.append(percentage_obs[i])
        obs.append(np.asarray(patient_obs))
    print("Sanity checks...")
    print("There are {} observations".format(len(obs)))
    print("First observation has shape", obs[0].shape)


    def interpolate_datapoint(obs):
        """

        :param obs: (n_obs, Dy + 1), [:, 0] records t of all obs
        :return:
        obs: (time, Dy)
        mask: (time, )
        time_interval: (time, )
        """
        days = obs[:, 0].astype(int)
        time = days[-1] - days[0] + 1

        mask = np.ones((time, ), dtype=bool)
        time_interval = np.zeros((time, ))

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

        return interpolated_obs, mask, time_interval


    def interpolate_data(obs_train):

        interpolated_obs_train = []

        mask_train = []

        time_interval_train = []

        for obs in obs_train:
            obs, mask, time_interval = interpolate_datapoint(obs)

            interpolated_obs_train.append(obs)

            mask_train.append(mask)
            time_interval_train.append(time_interval)


        return interpolated_obs_train, mask_train, time_interval_train

    new_obs, masks, time_interval = interpolate_data(obs)


    def bar_plot(ax, obs, mask, to_normalize=True):
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

    if not os.path.exists(viz_dir):
        os.mkdir(viz_dir)

    print("Saving plots to", viz_dir)

    for i in range(len(new_obs)):
        plt.figure()
        ax = plt.subplot(1,1,1)
        bar_plot(ax, new_obs[i], masks[i])
        plt.savefig("visualizations/sample_{}".format(i))
        plt.close()

    print("finished!")


if __name__ == '__main__':
    generate_visualizations()


