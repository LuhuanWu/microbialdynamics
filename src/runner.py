import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

# for data saving stuff
import joblib
# import git

# import from files
from src.model import SSM
from src.trainer import trainer

from src.SMC.SVO import SVO
from src.SMC.PSVO import PSVO
from src.SMC.IWAE import IWAE
from src.SMC.AESMC import AESMC

from src.rslts_saving.rslts_saving import *
from src.rslts_saving.fhn_rslts_saving import *
from src.rslts_saving.lorenz_rslts_saving import *

from src.utils.data_generator import generate_dataset
from src.utils.available_data import DATA_DIR_DICT, PERCENTAGE_DATA_DICT, COUNT_DATA_DICT
from src.utils.data_loader import load_data
from src.utils.data_interpolation import interpolate_data

def main(_):
    FLAGS = tf.app.flags.FLAGS

    # ========================================= parameter part begins ========================================== #
    Dx = FLAGS.Dx
    print_freq = FLAGS.print_freq

    # evaluation parameters
    if FLAGS.emission == "dirichlet" or FLAGS.emission == "mvn":
        y_hat_bar_plot_to_normalize = False
        if FLAGS.emission == "mvn":
            assert FLAGS.data_type in PERCENTAGE_DATA_DICT, "mvn emission is only compatible to percentage data."
    elif FLAGS.emission == "poisson" or FLAGS.emission == "multinomial":
        y_hat_bar_plot_to_normalize = True
    else:
        raise ValueError("Unsupported emission!")

    training_sample_idx = [int(x) for x in FLAGS.training_sample_idx.split(",")]
    if training_sample_idx == [-1]:
        training_sample_idx = None

    if FLAGS.use_2_q:
        FLAGS.q_uses_true_X = False

    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # ============================================= dataset part ============================================= #
    # generate data from simulation
    if FLAGS.generate_training_data:
        raise ValueError("Cannot generate data set from simulation, please provide a dataset file")
    # load data from file
    else:
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # repo = git.Repo('.', search_parent_directories=True)
        # repo_dir = repo.working_tree_dir  # microbialdynamics

        data_dir = DATA_DIR_DICT[FLAGS.data_type]
        data_dir = os.path.join(repo_dir, data_dir)

        if FLAGS.data_type == "toy":
            print("Use toy data")
            hidden_train, hidden_test, obs_train, obs_test, input_train, input_test = joblib.load(data_dir)
            print("Finish loading the toy data.")

            train_num = len(obs_train)
            T_train = obs_train[0].shape[0]
            test_num = len(obs_test)
            T_test = obs_test[0].shape[0]
            _mask_train = np.ones((train_num, T_train), dtype=bool)
            _mask_test = np.ones((test_num, T_test), dtype=bool)

            time_interval_train = np.zeros_like(_mask_train)
            time_interval_test = np.zeros_like(_mask_test)

            extra_inputs_train = np.zeros((train_num, T_train))
            extra_inputs_test = np.zeros((test_num, T_test))

        elif FLAGS.data_type in PERCENTAGE_DATA_DICT:
            hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, \
            extra_inputs_train, extra_inputs_test = \
                load_data(data_dir, Dx, FLAGS.isPython2, training_sample_idx=training_sample_idx)
            FLAGS.n_train, FLAGS.n_test = len(obs_train), len(obs_test)

            hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, \
            _mask_train, _mask_test, time_interval_train, time_interval_test, extra_inputs_train, extra_inputs_test = \
                interpolate_data(hidden_train, hidden_test, obs_train, obs_test, input_train, input_test,
                                 extra_inputs_train, extra_inputs_test, FLAGS.use_gp)

            if FLAGS.emission == "mvn":
                # transform to log additive ratio
               for i in range(len(obs_train)):
                    obs_train[i] = np.log(obs_train[i][:,:-1]) - np.log(obs_train[i][:, -1:])

               for i in range(len(obs_test)):
                   obs_test[i] = np.log(obs_test[i][:, :-1]) - np.log(obs_test[i][:, -1:])

        elif FLAGS.data_type in COUNT_DATA_DICT:
            hidden_train, hidden_test, obs_train, obs_test, input_train, input_test,\
            extra_inputs_train, extra_inputs_test = \
                load_data(data_dir, Dx, FLAGS.isPython2, training_sample_idx=training_sample_idx)
            FLAGS.n_train, FLAGS.n_test = len(obs_train), len(obs_train)

            hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, \
            _mask_train, _mask_test, time_interval_train, time_interval_test, extra_inputs_train, extra_inputs_test = \
                interpolate_data(hidden_train, hidden_test, obs_train, obs_test, input_train, input_test,
                                 extra_inputs_train, extra_inputs_test, FLAGS.use_gp)

        else:
            raise ValueError("Data type must be one of available data types.")

        if FLAGS.use_mask:
            mask_train, mask_test = _mask_train, _mask_test
        else:
            # set all the elements of mask to be True
            mask_train = []
            for m in _mask_train:
                mask_train.append(np.ones_like(m, dtype=bool))

            mask_test = []
            for m in _mask_test:
                mask_test.append(np.ones_like(m, dtype=bool))

    # clip saving_test_num to avoid it > n_train or n_test
    min_time_train = min([obs.shape[0] for obs in obs_train])
    min_time_test = min([obs.shape[0] for obs in obs_test])
    min_time = min(min_time_train, min_time_test)
    FLAGS.MSE_steps = min(FLAGS.MSE_steps, min_time - 2)
    FLAGS.saving_train_num = min(FLAGS.saving_train_num, FLAGS.n_train)
    FLAGS.saving_test_num = min(FLAGS.saving_test_num, FLAGS.n_test)

    print("finished preparing dataset")

    # ============================================== model part ============================================== #
    SSM_model = SSM(FLAGS)

    # at most one of them can be set to True
    assert FLAGS.PSVO + FLAGS.SVO + FLAGS.AESMC + FLAGS.IWAE < 2

    # SMC class to calculate loss
    if FLAGS.PSVO:
        SMC_train = PSVO(SSM_model, FLAGS)
    elif FLAGS.SVO:
        SMC_train = SVO(SSM_model, FLAGS)
    elif FLAGS.AESMC:
        SMC_train = AESMC(SSM_model, FLAGS)
    elif FLAGS.IWAE:
        SMC_train = IWAE(SSM_model, FLAGS)
    else:
        raise ValueError("Choose one of objectives among: PSVO, SVO, AESMC, IWAE")

    # at most of of them can be set to True
    assert FLAGS.log_dynamics + FLAGS.lar_dynamics < 2

    # =========================================== data saving part =========================================== #
    # create dir to save results
    Experiment_params = {"np":            FLAGS.n_particles,
                         "lr":            FLAGS.lr,
                         "epoch":         FLAGS.epoch,
                         "seed":          FLAGS.seed,
                         "rslt_dir_name": FLAGS.rslt_dir_name}

    RLT_DIR = create_RLT_DIR(Experiment_params)
    save_experiment_param(RLT_DIR, FLAGS)
    print("RLT_DIR:", RLT_DIR)

    # ============================================= training part ============================================ #
    mytrainer = trainer(SSM_model, SMC_train, FLAGS)
    mytrainer.init_data_saving(RLT_DIR)

    history, log = mytrainer.train(obs_train, obs_test,
                                   hidden_train, hidden_test,
                                   input_train, input_test,
                                   mask_train, mask_test,
                                   time_interval_train, time_interval_test,
                                   extra_inputs_train, extra_inputs_test,
                                   print_freq)

    # ======================================== final data saving part ======================================== #
    with open(RLT_DIR + "history.json", "w") as f:
        json.dump(history, f, indent=4, cls=NumpyEncoder)

    Xs, y_hat = log["Xs"], log["y_hat_original"]
    Xs_val = mytrainer.evaluate(Xs, mytrainer.test_feed_dict)

    y_hat_val_train = mytrainer.evaluate(y_hat, mytrainer.train_feed_dict)
    y_hat_val_test = mytrainer.evaluate(y_hat, mytrainer.test_feed_dict)
    print("finish evaluating training results")

    if FLAGS.emission == "mvn":
        # transform log additive ratio back to observation

        for i in range(len(obs_test)):
            obs_test[i] = np.exp(obs_test[i])
            p11 = 1 / (1 + obs_test[i].sum(axis=-1, keepdims=True))  # (n_days, 1)
            obs_test[i] = p11 * obs_test[i]

        for i in range(len(y_hat_val_test)):
            # y hat val = (batch_size, n_days, Dy)
            for j in range(len(y_hat_val_test[i])):
                y_hat_val_test[i][j] = np.exp(y_hat_val_test[i][j])
                p11 = 1 / (1 + y_hat_val_test[i][j].sum(axis=-1, keepdims=True))  # (n_days, 1)
                y_hat_val_test[i][j] = p11 * y_hat_val_test[i][j]

    plot_y_hat(RLT_DIR + "y_hat_train_plots", y_hat_val_train, obs_train, mask=mask_train,
               saving_num=FLAGS.saving_train_num)
    plot_y_hat(RLT_DIR + "y_hat_test_plots", y_hat_val_test, obs_test, mask=mask_test, saving_num=FLAGS.saving_test_num)

    plot_y_hat_bar_plot(RLT_DIR+"train_obs_y_hat_bar_plots", y_hat_val_train, obs_train, mask=mask_train,
                        saving_num=FLAGS.saving_train_num, to_normalize=y_hat_bar_plot_to_normalize)
    plot_y_hat_bar_plot(RLT_DIR+"test_obs_y_hat_bar_plots", y_hat_val_test, obs_test, mask=mask_test,
                        saving_num=FLAGS.saving_test_num, to_normalize=y_hat_bar_plot_to_normalize)

    if Dx == 2:
        plot_fhn_results(RLT_DIR, Xs_val)
    if Dx == 3:
        plot_lorenz_results(RLT_DIR, Xs_val)

    testing_data_dict = {"hidden_test": hidden_test[0:FLAGS.saving_test_num],
                         "obs_test": obs_test[0:FLAGS.saving_test_num],
                         "input_test": input_test[0:FLAGS.saving_test_num]}
    
    learned_model_dict = {"Xs_val": Xs_val,
                          "y_hat_val_test": y_hat_val_test}
    data_dict = {"testing_data_dict": testing_data_dict,
                 "learned_model_dict": learned_model_dict}
    
    with open(RLT_DIR + "data.p", "wb") as f:
        pickle.dump(data_dict, f)

    plot_MSEs(RLT_DIR, history["MSE_trains"], history["MSE_tests"], print_freq)
    plot_R_square(RLT_DIR, history["R_square_trains"], history["R_square_tests"], print_freq)
    plot_log_ZSMC(RLT_DIR, history["log_ZSMC_trains"], history["log_ZSMC_tests"], print_freq)

    print("finish plotting!")

