# available data options
toy_data_dir = "data/fhn_with_inputs_dirichlet"

percentage_data_dir = "data/microbio.p"
count_data_dir = "data/count_microbio.p"

percentage_noinputs_dir = "data/microbio_noinputs.p"
count_noinputs_dir = "data/count_microbio_noinputs.p"

pink_count_data_dir = "data/pink_count_microbio.p"
cyan_count_data_dir = "data/cyan_count_microbio.p"

clv_percentage_Dx_1_data_dir = "data/clv/data/clv_ntrain_600_Dx_1.p"
clv_percentage_Dx_2_data_dir = "data/clv/data/clv_ntrain_600_Dx_2.p"
clv_percentage_Dx_3_data_dir = "data/clv/data/clv_ntrain_600_Dx_3.p"
clv_percentage_Dx_4_data_dir = "data/clv/data/clv_ntrain_600_Dx_4.p"
clv_percentage_Dx_5_data_dir = "data/clv/data/clv_ntrain_600_Dx_5.p"
clv_percentage_Dx_6_data_dir = "data/clv/data/clv_ntrain_600_Dx_6.p"
clv_percentage_Dx_7_data_dir = "data/clv/data/clv_ntrain_600_Dx_7.p"
clv_percentage_Dx_8_data_dir = "data/clv/data/clv_ntrain_600_Dx_8.p"
clv_percentage_Dx_9_data_dir = "data/clv/data/clv_ntrain_600_Dx_9.p"
clv_percentage_Dx_10_data_dir = "data/clv/data/clv_ntrain_600_Dx_10.p"
clv_percentage_Dx_10_ntrain_1000_data_dir = "data/clv/data/clv_ntrain_1000_Dx_10.p"
clv_percentage_Dx_10_ntrain_1200_data_dir = "data/clv/data/clv_ntrain_1200_Dx_10.p"

clv_count_Dx_1_data_dir = "data/clv/data/clv_count_ntrain_600_Dx_1.p"
clv_count_Dx_2_data_dir = "data/clv/data/clv_count_ntrain_600_Dx_2.p"
clv_count_Dx_3_data_dir = "data/clv/data/clv_count_ntrain_600_Dx_3.p"
clv_count_Dx_4_data_dir = "data/clv/data/clv_count_ntrain_600_Dx_4.p"
clv_count_Dx_5_data_dir = "data/clv/data/clv_count_ntrain_600_Dx_5.p"
clv_count_Dx_6_data_dir = "data/clv/data/clv_count_ntrain_600_Dx_6.p"
clv_count_Dx_7_data_dir = "data/clv/data/clv_count_ntrain_600_Dx_7.p"
clv_count_Dx_8_data_dir = "data/clv/data/clv_count_ntrain_600_Dx_8.p"
clv_count_Dx_9_data_dir = "data/clv/data/clv_count_ntrain_600_Dx_9.p"
clv_count_Dx_10_data_dir = "data/clv/data/clv_count_ntrain_600_Dx_10.p"
clv_count_Dx_10_ntrain_1000_data_dir = "data/clv/data/clv_count_ntrain_1000_Dx_10.p"
clv_count_Dx_10_ntrain_1200_data_dir = "data/clv/data/clv_count_ntrain_1200_Dx_10.p"

DATA_DIR_DICT = dict(toy=toy_data_dir,
                     percentage=percentage_data_dir, percentage_noinputs=percentage_noinputs_dir,
                     count=count_data_dir, count_noinputs=count_noinputs_dir,
                     pink_count=pink_count_data_dir, cyan_count=cyan_count_data_dir,
                     clv_percentage_Dx_1=clv_percentage_Dx_1_data_dir,
                     clv_percentage_Dx_2=clv_percentage_Dx_2_data_dir,
                     clv_percentage_Dx_3=clv_percentage_Dx_3_data_dir,
                     clv_percentage_Dx_4=clv_percentage_Dx_4_data_dir,
                     clv_percentage_Dx_5=clv_percentage_Dx_5_data_dir,
                     clv_percentage_Dx_6=clv_percentage_Dx_6_data_dir,
                     clv_percentage_Dx_7=clv_percentage_Dx_7_data_dir,
                     clv_percentage_Dx_8=clv_percentage_Dx_8_data_dir,
                     clv_percentage_Dx_9=clv_percentage_Dx_9_data_dir,
                     clv_percentage_Dx_10=clv_percentage_Dx_10_data_dir,
                     clv_percentage_Dx_10_ntrain_1000=clv_percentage_Dx_10_ntrain_1000_data_dir,
                     clv_percentage_Dx_10_ntrain_1200=clv_percentage_Dx_10_ntrain_1200_data_dir,
                     clv_count_Dx_1=clv_count_Dx_1_data_dir,
                     clv_count_Dx_2=clv_count_Dx_2_data_dir,
                     clv_count_Dx_3=clv_count_Dx_3_data_dir,
                     clv_count_Dx_4=clv_count_Dx_4_data_dir,
                     clv_count_Dx_5=clv_count_Dx_5_data_dir,
                     clv_count_Dx_6=clv_count_Dx_6_data_dir,
                     clv_count_Dx_7=clv_count_Dx_7_data_dir,
                     clv_count_Dx_8=clv_count_Dx_8_data_dir,
                     clv_count_Dx_9=clv_count_Dx_9_data_dir,
                     clv_count_Dx_10=clv_count_Dx_10_data_dir,
                     clv_count_Dx_10_ntrain_1000=clv_count_Dx_10_ntrain_1000_data_dir,
                     clv_count_Dx_10_ntrain_1200=clv_count_Dx_10_ntrain_1200_data_dir,
                     )

PERCENTAGE_DATA_TYPE = ["percentage", "percentage_noinputs",
                        "clv_percentage_Dx_1",
                        "clv_percentage_Dx_2",
                        "clv_percentage_Dx_3",
                        "clv_percentage_Dx_4",
                        "clv_percentage_Dx_5",
                        "clv_percentage_Dx_6",
                        "clv_percentage_Dx_7",
                        "clv_percentage_Dx_8",
                        "clv_percentage_Dx_9",
                        "clv_percentage_Dx_10",
                        "clv_percentage_Dx_10_ntrain_1000",
                        "clv_percentage_Dx_10_ntrain_1200"]

COUNT_DATA_TYPE = ["count", "count_noinputs", "pink_count", "cyan_count",
                   "clv_count_Dx_1",
                   "clv_count_Dx_2",
                   "clv_count_Dx_3",
                   "clv_count_Dx_4",
                   "clv_count_Dx_5",
                   "clv_count_Dx_6",
                   "clv_count_Dx_7",
                   "clv_count_Dx_8",
                   "clv_count_Dx_9",
                   "clv_count_Dx_10",
                   "clv_count_Dx_10_ntrain_1000",
                   "clv_count_Dx_10_ntrain_1200"]
