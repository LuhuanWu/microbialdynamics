# Microbial Dynamic

### Installation

The code is written for Python 3.6, all dependencies can be installed by

```
pip install -r requirements.txt
```

### To Run

```
cd src
python runner_flag.py
```

Results will be store in `src/rslts/clv_group/[datatype]_Dx[Dx]` where `[datatype]` and `[Dx]` are flags in `src/runner_flag.py`

### Data Format

a python 3 `pickle` file contains a `dict` with following items:

* `Ytrain`: (`list` of `np.array`) each `np.array`, named `y_i`, represents observations for a training trajectory. `y_i` has shape `(n_observed_days, 1 + Dy)`, and `y_i[:, 0]` is the date when the observation is recorded. `y_i[:, 1:]` are absolute abundance of taxa.
* `Ytest`: (`list` of `np.array`) observations of test set.
* `Vtrain`: (`list` of `np.array`) each `np.array`, named `v_i`, represents perturbations for a training trajectory. Similar to observations, `v_i` has shape `(n_perturbation_days, 1 + Dv)`, and `v_i[:, 0]` is the date when the perturbations is recorded. `v_i[:, 1:]` are absolute abundance of taxa. Currently, perturbations are either 1 or 0 (appear or not).
* `Vtest`: (`list` of `np.array`) perturbations of test set.
* `counts_train`: (`list` of `np.array`) each `np.array`, named `c_i`, represents total counts of each day for a training trajectory. `c_i` has shape `(n_observed_days)`.
* `counts_test`: total counts of test set.

The following items are optional:

* `Xtrain`: (`list` of `np.array`) each `np.array`, named `x_i`, represents hidden states for a training trajectory. `x_i` has shape `(n_total_days, Dx)`, and it's only know for simulation data.
* `Xtest`: (`list` of `np.array`)  hidden states of test set.
* `A`, `g`, `Wv`: simulation parameters for interaction, growth rate, and perturbation matrix.
* `k_var`, `k_len`: simulation parameters for added gaussian process noise.

### To add a new data file

1. prepare a `dict` meetings the above requirements and dump it to a python 3 `pickle` file.
2. add the file to `src/utils/available_data.py`
   * specifies its path, such as `group_Dx_2_Dv_0_ntrain_300_Kvar_05_dir = "data/clv_gp/group_interaction/Dx_2_Dy_8_Dv_0_s_1_Kvar_0.3_ntrain_300.p"`
   * add it to `COUNT_DATA_DICT`, such as  `group_Dx_2_Dv_0_ntrain_300_Kvar_05=group_Dx_2_Dv_0_ntrain_300_Kvar_05_dir`

