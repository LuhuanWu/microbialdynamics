from src.utils.data_loader import load_data
from src.utils.data_interpolation import interpolate_data
from src.rslts_saving.rslts_saving import plot_obs_bar_plot

datadir = "/Users/leah/Columbia/courses/19summer/microbialdynamics/data/count_microbio.p"
Dx = 3

hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, extra_inputs_train, extra_inputs_test = \
    load_data(datadir, Dx, False)

hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, _mask_train, _mask_test, time_interval_train, time_interval_test, extra_inputs_train, extra_input_test = \
                interpolate_data(hidden_train, hidden_test, obs_train, obs_test, input_train, input_test,
                                 extra_inputs_train, extra_inputs_test, False)


masks = _mask_train + _mask_test
obs = obs_train + obs_test

plot_obs_bar_plot(obs, masks, True)

