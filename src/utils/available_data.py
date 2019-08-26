# available data options
toy_data_dir = "data/fhn_with_inputs_dirichlet"

percentage_data_dir = "data/microbio.p"
count_data_dir = "data/count_microbio.p"

percentage_noinputs_dir = "data/microbio_noinputs.p"
count_noinputs_dir = "data/count_microbio_noinputs.p"

pink_count_data_dir = "data/pink_count_microbio.p"
cyan_count_data_dir = "data/cyan_count_microbio.p"

clv_data_dir = "data/clv.p"
clv_08_data_dir = "data/clv_data_w_missing_obs/clv_0.8_obs.p"
clv_06_data_dir = "data/clv_data_w_missing_obs/clv_0.6_obs.p"
clv_05_data_dir = "data/clv_data_w_missing_obs/clv_0.5_obs.p"
clv_04_data_dir = "data/clv_data_w_missing_obs/clv_0.4_obs.p"

clv_input_root_dir = "data/clv_data_with_input/"
clv_input_dir = clv_input_root_dir + "clv_percentage.p"
clv_input_08_dir = clv_input_root_dir + "clv_percentage_0.8_obs.p"
clv_input_06_dir = clv_input_root_dir + "clv_percentage_0.6_obs.p"
clv_input_05_dir = clv_input_root_dir + "clv_percentage_0.5_obs.p"
clv_input_04_dir = clv_input_root_dir + "clv_percentage_0.4_obs.p"

clv_input_noise_dir = "data/clv/clv_w_input_and_noise.p"

clv_count_dir = "data/count_clv.p"

DATA_DIR_DICT = dict(toy=toy_data_dir, percentage=percentage_data_dir,
                     count=count_data_dir,
                     percentage_noinputs=percentage_noinputs_dir,
                     count_noinputs=count_noinputs_dir,
                     pink_count=pink_count_data_dir, cyan_count=cyan_count_data_dir,
                     clv=clv_data_dir, clv_count=clv_count_dir,
                     clv_08=clv_08_data_dir, clv_06=clv_06_data_dir,
                     clv_05=clv_05_data_dir, clv_04=clv_04_data_dir,
                     clvi=clv_input_dir, clvi_08=clv_input_08_dir,
                     clvi_06=clv_input_06_dir, clvi_05=clv_05_data_dir, clvi_04=clv_input_04_dir,
                     clv_input_noise=clv_input_noise_dir)

PERCENTAGE_DATA_TYPE = ["percentage", "percentage_noinputs", "clv", "clv_08", "clv_06", "clv_05", "clv_04",
                        "clvi", "clvi_08", "clvi_06", "clvi_05", "clvi_04", "clv_input_noise"]

COUNT_DATA_TYPE = ["count", "count_noinputs", "pink_count", "cyan_count", "clv_count"]



