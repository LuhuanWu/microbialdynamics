import os
import copy
import pickle
import numpy as np

if __name__ == "__main__":
    fname = os.path.join("data", "clv_gp_Dx_5_Dy_10_Dv_0_ntrain_1500_obp_10_ls_2.p")
    with open(fname, "rb") as f:
        d = pickle.load(f)

    for n_train in [50, 100, 200, 500, 1000]:
        d_shorter = copy.deepcopy(d)
        d_shorter["Xtrain"] = d_shorter["Xtrain"][:n_train]
        d_shorter["Ytrain"] = d_shorter["Ytrain"][:n_train]
        d_shorter["Vtrain"] = d_shorter["Vtrain"][:n_train]
        d_shorter["counts_train"] = d_shorter["counts_train"][:n_train]
        f_shorter_name = fname.split("_")
        f_shorter_name[f_shorter_name.index("ntrain") + 1] = str(n_train)
        f_shorter_name = "_".join(f_shorter_name)
        with open(f_shorter_name, "wb") as f:
            pickle.dump(d_shorter, f)
