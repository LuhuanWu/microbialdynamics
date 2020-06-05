import os
import csv
import biom
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_theta(tree_str):
    tree_str = tree_str[:-1]  # remove semi-colon
    start = 0
    new_str = ""
    while start < len(tree_str):
        for i in range(start, len(tree_str)):
            char = tree_str[i]
            if char in "(),:":
                break
        word = tree_str[start:i]
        if is_int(word) and start > 0 and tree_str[start - 1] in "(,":
            new_str += word
        if char in "(),":
            new_str += char
        start = i + 1

    from ast import literal_eval
    tree = literal_eval(new_str)

    def get_theta_helper(tree_):
        if isinstance(tree_, int):
            return None
        left_theta = get_theta_helper(tree_[1])
        right_theta = get_theta_helper(tree_[0])
        if left_theta is None and right_theta is None:
            return np.array([[+1, -1]])
        if left_theta is None:
            left_theta_ = np.array([[+1]])
        else:
            left_theta_ = left_theta
        if right_theta is None:
            right_theta_ = np.array([[+1]])
        else:
            right_theta_ = right_theta

        row = np.concatenate([np.ones(left_theta_.shape[1]), -np.ones(right_theta_.shape[1])])
        left_zeros = np.zeros((left_theta_.shape[0], right_theta_.shape[1]))
        right_zeros = np.zeros((right_theta_.shape[0], left_theta_.shape[1]))
        left_theta_ = np.concatenate([left_theta_, left_zeros], axis=-1)
        right_theta_ = np.concatenate([right_zeros, right_theta_], axis=-1)

        if left_theta is None:
            theta = np.concatenate([row[None, :], right_theta_], axis=0)
        elif right_theta is None:
            theta = np.concatenate([row[None, :], left_theta_], axis=0)
        else:
            theta = np.concatenate([row[None, :], left_theta_, right_theta_], axis=0)
        assert theta.shape[0] + 1 == theta.shape[1]
        return theta

    theta = get_theta_helper(tree)

    # add "all other" to theta
    new_root_row = np.concatenate([np.ones(theta.shape[1]), [-1]])
    theta = np.concatenate([theta, np.zeros((theta.shape[0], 1))], axis=-1)
    theta = np.concatenate([new_root_row[None, :], theta], axis=0)

    return theta


def bar_plot(ax, obs, mask, to_normalize=True):
    if to_normalize:
        obs = obs / np.sum(obs, axis=-1, keepdims=True)

    time, Dy = obs.shape

    # make missing obs = 0
    masked_obs = np.zeros_like(obs)
    masked_obs[mask] = obs[mask]

    ax.set_ylabel("Relative Abundance")
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
    ax.set_ylabel("Perturbation")
    ax.set_yticks(np.arange(Dv) + 0.5)
    ax.set_yticklabels(["drink", "eat", "sleep"])
    sns.despine()


def count_plot(ax, obs, mask):
    time, Dy = obs.shape

    # make missing obs = 0
    masked_obs = np.zeros_like(obs)
    masked_obs[mask] = obs[mask]

    count = masked_obs.sum(axis=-1)

    ax.set_xlabel("Time")
    ax.set_ylabel("Depth")
    ax.bar(np.arange(time), count, edgecolor='white')
    ax.set_xticks(np.arange(time))
    sns.despine()


def plot_inputs_and_obs(inputs, obs, masks, i, to_normalize=True):
    has_inputs = inputs[0].shape[1] > 0
    if has_inputs:
        plt.figure(figsize=(20, 12))

        ax1 = plt.subplot(3, 1, 1)
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        input_plot(ax1, inputs[i])
        ax1.grid(axis='x')
    else:
        plt.figure(figsize=(20, 8))
        ax2 = plt.subplot(2, 1, 1)
        ax3 = plt.subplot(2, 1, 2, sharex=ax2)

    bar_plot(ax2, obs[i], masks[i], to_normalize=to_normalize)
    ax2.grid(axis='x')

    count_plot(ax3, obs[i], masks[i])
    ax3.grid(axis='x')

    os.makedirs("plots/data", exist_ok=True)
    plt.savefig("plots/data/{}".format(i))
    plt.close("all")


def relative_abundance_plot(data_sum):
    sorted_data_sum = np.array(-np.sort(-data_sum))
    sorted_data_sum = sorted_data_sum[:100]
    plt.figure()
    ax1 = plt.gca()
    ax1.bar(np.arange(len(sorted_data_sum)), sorted_data_sum)
    ax1.set_ylabel("relative abundance")
    cum = [0]
    for ele in sorted_data_sum:
        cum.append(cum[-1] + ele)
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(cum)), cum, 'r')
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("cumulative relative abundance")
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/taxa_abundance")
    # plt.show()


def get_traj_and_input(data, sel_idxes):
    trajs, inputs = [], []
    traj, Input, timestamps, prev_time_idx, prev_timestamp = [], [], [], None, None
    others = []
    foods = []
    current_host_id = None

    tmp = []
    tmp_ = {}

    with open("map.onlymedtime.txt") as tsv:
        tsv_reader = csv.DictReader(tsv, delimiter="\t")
        for i, line in enumerate(tsv_reader):
            sampled_id = line["#SampleID"]
            host_id = line["hostid"]
            conttime = int(line["conttime"])

            drink = line["drink"]
            eat = line["eat"]
            sleep = line["sleep"]
            other = line["other"]

            tmp_[line["body site"]] = tmp_.get(line["body site"], 0) + 1

            if sampled_id not in sample_ids:
                # print("host {}'s sample {} at time {} doesn't exist".format(host_id, sampled_id, line["dayhourmin"]))
                continue
            if line["body site"] != "saliva":
                continue

            sample_interval = 15
            time_idx = conttime // sample_interval
            traj_terminate = False

            if host_id != current_host_id:
                traj_terminate = True
                # print("host_id:", host_id)

            if prev_time_idx is not None:
                cur_timestamp = line["dayhourmin"]
                if time_idx == prev_time_idx:
                    # print("host {} had two samples at time {} and {}, ignore the 2nd one".format(
                    #     host_id, prev_timestamp, cur_timestamp))
                    continue
                elif time_idx - prev_time_idx > 2 * 60 / sample_interval and line["hour"]:
                    # print(prev_timestamp, cur_timestamp)
                    traj_terminate = True
            prev_time_idx, prev_timestamp = time_idx, line["dayhourmin"]

            if traj_terminate:
                if len(traj) > 4:
                    traj, Input, timestamps = np.array(traj), np.array(Input), np.array(timestamps) - timestamps[0]
                    traj = np.concatenate([timestamps[:, None], traj], axis=-1)
                    Input = np.concatenate([timestamps[:-1, None], Input[1:, :]], axis=-1)
                    trajs.append(traj)
                    inputs.append(Input)
                current_host_id = host_id
                traj, Input, timestamps, prev_time_idx, prev_timestamp = [], [], [], None, None

            # observations (counts)
            assert len(np.where(sample_ids == sampled_id)[0]) == 1
            row_idx = np.where(sample_ids == sampled_id)[0][0]
            sel_taxon_counts = data[row_idx][sel_idxes]                    # (num_taxa,)
            remain_counts = data[row_idx].sum() - sel_taxon_counts.sum()   # scalar
            y = np.concatenate([sel_taxon_counts, [remain_counts]])
            traj.append(y)

            # inputs [drink, eat, sleep] all 0/1
            Input.append([drink != "", eat != "", sleep != ""])

            timestamps.append(time_idx)

            if other != "":
                others.append(other)
            if eat != "":
                foods.append(eat)

    if traj is not None:
        traj, Input, timestamps = np.array(traj), np.array(Input), np.array(timestamps)
        traj = np.concatenate([timestamps[:, None], traj], axis=-1)
        Input = np.concatenate([timestamps[:-1, None], Input[1:, :]], axis=-1)
        trajs.append(traj)
        inputs.append(Input)

    # print(set(tmp))
    # print(tmp_)

    # print("\nfoods includes:")
    # for food in set(foods):
    #     print(food)

    # print("\nothers includes:")
    # for other in set(others):
    #     print(other)

    return trajs, inputs


def save_data(trajs, inputs):
    idx = list(range(len(trajs)))
    n_train = int(len(trajs) * 0.8)
    total_num = np.sum([len(traj) for traj in trajs])
    while True:
        np.random.shuffle(idx)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        train_num = np.sum([len(trajs[idx]) for idx in train_idx])
        test_num = np.sum([len(trajs[idx]) for idx in test_idx])
        if 0.75 <= train_num / total_num <= 0.8:
            break

    c_data = {}
    c_data['theta'] = theta
    c_data['Ytrain'] = [trajs[idx] for idx in train_idx]  # count
    c_data['Ytest'] = [trajs[idx] for idx in test_idx]
    c_data['Vtrain'] = [inputs[idx] for idx in train_idx]
    c_data['Vtest'] = [inputs[idx] for idx in test_idx]

    with open("oral_medium.p", "wb") as f:
        pickle.dump(c_data, f)

    # stats
    print("Num of trajs", len(trajs))
    print("Num of total timestamps", total_num)
    print("Num of train timestamps", train_num)
    print("Num of test timestamps", test_num)


def plot_data(trajs, inputs, plot_num=-1):
    import sys
    sys.path.append("../..")
    from microbialdynamics.src.utils.data_interpolation import interpolate_data

    obs_train, input_train = trajs, inputs
    obs_test, input_test = [], []
    hidden_train = [None for _ in obs_train]
    hidden_test = [None for _ in obs_test]
    extra_inputs_train = [obs[:, 1:].sum(axis=-1) for obs in obs_train]
    extra_inputs_test = [None for _ in obs_test]

    hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, \
        _mask_train, _mask_test, time_interval_train, time_interval_test, extra_inputs_train, extra_input_test = \
        interpolate_data(hidden_train, hidden_test, obs_train, obs_test, input_train, input_test,
                         extra_inputs_train, extra_inputs_test, interpolation_type=None)

    masks = _mask_train + _mask_test
    obs = obs_train + obs_test
    inputs = input_train + input_test

    if plot_num == -1:
        plot_num = len(obs)
    for i in range(0, plot_num):
        plot_inputs_and_obs(inputs, obs, masks, i)


if __name__ == "__main__":
    table = biom.load_table("all.biom")
    data = table._data.toarray().T

    sample_ids = table.ids()
    taxon_ids = table.ids(axis="observation")

    # filter to top N taxa
    num_taxa = 20
    data_norm = data / data.sum(axis=-1, keepdims=True)
    data_sum = np.sum(data_norm, axis=0) / np.sum(data_norm)
    sel_idxes = [idx for i, idx in enumerate(np.argsort(-data_sum)) if i < num_taxa]
    sel_taxon_ids = taxon_ids[sel_idxes]

    # save sequence to FASTA file
    file_str = ""
    line_len = 50
    for idx, taxon_id in zip(sel_idxes, sel_taxon_ids):
        file_str += ">{}\n".format(idx)
        for start in range(0, len(taxon_id), line_len):
            file_str += "{}\n".format(taxon_id[start:start + line_len])
    os.makedirs("ph_tree", exist_ok=True)
    with open("ph_tree/top_{}_taxon.fasta".format(num_taxa), "w") as f:
        f.write(file_str)

    # check relative abundance
    relative_abundance_plot(data_sum)

    # get trajs and inputs
    trajs, inputs = get_traj_and_input(data, sel_idxes)

    # theta
    with open("ph_tree/top_{}_taxon.fasta.final_tree.nw".format(num_taxa), "r") as f:
        tree_str = f.read()
    theta = get_theta(tree_str)

    # save data
    save_data(trajs, inputs)

    # visualization
    plot_data(trajs, inputs)
