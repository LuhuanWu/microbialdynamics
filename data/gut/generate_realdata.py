import numpy as np
import pandas as pd
import os
import pickle


use_split = False
split_threshold = 3
k = 2

file1 = "taur-events-0925.csv"
file2 = "taur-otu-table-15tpts-0925.csv"
f1 = pd.read_csv(file1)
f2 = pd.read_csv(file2, header=None)
data1 = f1.to_numpy()
data2 = f2.to_numpy()
pid = np.array(f1['patientID'])

# obs
measure_pid = np.array(f2.loc[0, 1:], dtype=int)
f2pid = np.unique(measure_pid)
print("{} patients with id {}".format(len(f2pid), f2pid))

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

dates = data2[1, 1:].astype(int)
count_obs = np.zeros((count.shape[0], k + 2))
count_obs[:, 0] = dates
count_obs[:, 1:-1] = count[:, top_k_spieces_idx]
count_obs[:, -1] = count.sum(axis=-1) - count[:, top_k_spieces_idx].sum(axis=-1)
percentage_obs = np.copy(count_obs)
percentage_obs[:, 1:-1] = count_obs[:, 1:-1] / np.sum(count_obs[:, 1:-1], axis=1, keepdims=True)


Y = []
percentage_Y = []
obs_pid = []

pid_measure_start = 0
pid = measure_pid[0]
while True:
    patient_obs = []
    patient_percentage_obs = []
    for i in range(pid_measure_start, len(measure_pid)):
        if measure_pid[i] != pid:
            pid_measure_start = i
            pid = measure_pid[i]
            break
        if use_split and i > pid_measure_start and dates[i] - dates[i - 1] > 8:
            pid_measure_start = i
            break
        patient_obs.append(count_obs[i])
        patient_percentage_obs.append(percentage_obs[i])
    if len(patient_obs) >= split_threshold:
        Y.append(np.asarray(patient_obs))
        percentage_Y.append(np.asarray(patient_percentage_obs))
        obs_pid.append(pid)
    if i == len(measure_pid) - 1:
        break

obs, percentage_obs = Y, percentage_Y

# inputs

event_pid, event, event_start, event_end = data1.T
event_pid = np.array(event_pid, dtype=int)
event_start = np.array(event_start, dtype=int)
event_end = np.array(event_end, dtype=int)
unique_event = list(np.unique(event))
num_event = len(unique_event)
print("total {} kinds of events: {}".format(num_event, unique_event))

Input = []
for pobs, pid in zip(obs, obs_pid):
    patient_event_idxs = np.where(event_pid == pid)[0]
    obs_start = int(pobs[0, 0])
    obs_end = int(pobs[-1, 0])
    patient_input = np.zeros((obs_end - obs_start + 1, num_event + 1))
    patient_input[:, 0] = np.arange(obs_start, obs_end + 1)

    for event_idx in patient_event_idxs:
        patient_event = event[event_idx]
        event_id = unique_event.index(patient_event) + 1
        if patient_event == "surgery":
            surgery_start = min(patient_input.shape[0], max(0, 0 - obs_start))
            patient_input[surgery_start:, event_id] = np.ones(patient_input.shape[0] - surgery_start)
        else:
            for i in range(event_start[event_idx], event_end[event_idx] + 1):
                if not obs_start <= i <= obs_end:
                    continue
                patient_input[i - obs_start, event_id] = 1.0
    Input.append(patient_input)

# counts
counts = []
for single_obs in obs:
    single_counts = single_obs[:, 1:].sum(axis=-1)
    counts.append(single_counts)

# save
n_train = int(0.8 * len(obs))
data = {}
data["Ytrain"] = obs[:n_train]
data["Ytest"] = obs[n_train:]
data["Vtrain"] = Input[:n_train]
data["Vtest"] = Input[n_train:]
data["counts_train"] = counts[:n_train]
data["counts_test"] = counts[n_train:]

count_fname = "count_microbio{}_k{}.p".format("_split_{}".format(split_threshold) if use_split else "", k)
percentage_fname = "microbio{}_{}.p".format("_split_{}".format(split_threshold) if use_split else "", k)
with open(count_fname, "wb") as f:
    pickle.dump(data, f)
data["Ytrain"] = percentage_obs[:n_train]
data["Ytest"] = percentage_obs[n_train:]
with open(percentage_fname, "wb") as f:
    pickle.dump(data, f)
