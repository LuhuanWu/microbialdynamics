# available data options
toy_data_dir = "data/fhn_with_inputs_dirichlet"

# --------------------------------------------- real data --------------------------------------------- #
percentage_data_dir = "data/microbio.p"
count_data_dir = "data/count_microbio.p"

# k6
count_data_k2_dir = "data/count_microbio_k2.p"
count_data_k6_dir = "data/count_microbio_k6.p"
count_data_k8_dir = "data/count_microbio_k8.p"

percentage_small_dir = "data/microbio_small.p"
count_small_dir = "data/count_microbio_small.p"

# ---------------------------------------------- clv data --------------------------------------------- #
clv_count_Dx10_obs02_dir = "data/clv/data/clv_count_ntrain_200_Dx_10_obs_02.p"
clv_count_Dx10_obs06_dir = "data/clv/data/clv_count_ntrain_200_Dx_10_obs_06.p"
clv_count_Dx10_obs10_dir = "data/clv/data/clv_count_ntrain_200_Dx_10_obs_10.p"

clv_count_Dx10_obs02_noinput_dir = "data/clv/data/clv_count_ntrain_200_Dx_10_obs_02_noinput.p"
clv_count_Dx10_obs06_noinput_dir = "data/clv/data/clv_count_ntrain_200_Dx_10_obs_06_noinput.p"
clv_count_Dx10_obs10_noinput_dir = "data/clv/data/clv_count_ntrain_200_Dx_10_obs_10_noinput.p"

clv_count_Dx4_obs10_noinput_dir = "data/clv/data/clv_count_ntrain_20_Dx_4_obs_10_noinput.p"

# ----------------------------------- lda data -------------------------------------------------------- #
# noinput
lda_4groups_8taxons_dir = "data/lda/4groups_8taxons.p"
lda_4groups_8taxons_full_dir = "data/lda/4groups_8taxons_full.p"

# ----------------------------------------- interpolation data ---------------------------------------- #
count_clv_interpolation_data_dir = "data/interpolation/count_clv.p"

PERCENTAGE_DATA_DICT = dict(percentage=percentage_data_dir,
                            percentage_small=percentage_small_dir
                            )


COUNT_DATA_DICT = dict(count=count_data_dir,
                       count_k2=count_data_k2_dir,
                       count_k6=count_data_k6_dir,
                       count_k8=count_data_k8_dir,
                       count_small=count_small_dir,
                       clv_count_Dx10_obs02=clv_count_Dx10_obs02_dir,
                       clv_count_Dx10_obs06=clv_count_Dx10_obs06_dir,
                       clv_count_Dx10_obs10=clv_count_Dx10_obs10_dir,
                       clv_count_Dx10_obs02_noinput=clv_count_Dx10_obs02_noinput_dir,
                       clv_count_Dx10_obs06_noinput=clv_count_Dx10_obs06_noinput_dir,
                       clv_count_Dx10_obs10_noinput=clv_count_Dx10_obs10_noinput_dir,
                       clv_count_Dx4_obs10_noinput=clv_count_Dx4_obs10_noinput_dir,
                       lda_4groups_8taxons=lda_4groups_8taxons_dir,
                       lda_4groups_8taxons_full=lda_4groups_8taxons_full_dir
                       )



INTERPOLATION_DATA_DICT = dict(count_clv=count_clv_interpolation_data_dir,)

DATA_DIR_DICT = {**PERCENTAGE_DATA_DICT, **COUNT_DATA_DICT}
DATA_DIR_DICT["toy"] = toy_data_dir
