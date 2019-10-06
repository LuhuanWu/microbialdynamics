import numpy as np
from scipy.special import logsumexp

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
from src.utils.available_data import DATA_DIR_DICT, PERCENTAGE_DATA_DICT, COUNT_DATA_DICT, INTERPOLATION_DATA_DICT
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

    FLAGS.epochs = [int(epoch) for epoch in FLAGS.epochs.split(",")]

    training_sample_idx = [int(x) for x in FLAGS.training_sample_idx.split(",")]
    test_sample_idx = [int(x) for x in FLAGS.test_sample_idx.split(",")]
    if training_sample_idx == [-1]:
        training_sample_idx = None
    if test_sample_idx == [-1]:
        test_sample_idx = None

    if FLAGS.use_2_q:
        FLAGS.q_uses_true_X = False

    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # ============================================= dataset part ============================================= #
    # generate data from simulation
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # repo = git.Repo('.', search_parent_directories=True)
    # repo_dir = repo.working_tree_dir  # microbialdynamics

    data_dir = DATA_DIR_DICT[FLAGS.data_type]
    data_dir = os.path.join(repo_dir, data_dir)
    if FLAGS.interpolation_data_type == "placeholder":
        interpolation_data = None
    else:
        interpolation_data_dir = INTERPOLATION_DATA_DICT[FLAGS.interpolation_data_type]
        interpolation_data_dir = os.path.join(repo_dir, interpolation_data_dir)
        with open(interpolation_data_dir, "rb") as f:
            interpolation_data = pickle.load(f)

    if FLAGS.data_type == "toy":
        print("Use toy data")
        hidden_train, hidden_test, obs_train, obs_test, input_train, input_test = joblib.load(data_dir)
        print("Finish loading the toy data.")

        train_num, test_num = len(obs_train), len(obs_test)
        T_train, T_test = obs_train[0].shape[0], obs_test[0].shape[0]

        _mask_train, _mask_test = np.ones((train_num, T_train), dtype=bool), np.ones((test_num, T_test), dtype=bool)
        time_interval_train, time_interval_test = np.zeros_like(_mask_train), np.zeros_like(_mask_test)
        extra_inputs_train, extra_inputs_test = np.zeros((train_num, T_train)), np.zeros((test_num, T_test))

    elif FLAGS.data_type in PERCENTAGE_DATA_DICT or FLAGS.data_type in COUNT_DATA_DICT:
        hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, \
        extra_inputs_train, extra_inputs_test = \
            load_data(data_dir, Dx, FLAGS.isPython2,
                      training_sample_idx=training_sample_idx, test_sample_idx=test_sample_idx)
        FLAGS.n_train, FLAGS.n_test = len(obs_train), len(obs_test)

        hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, \
        _mask_train, _mask_test, time_interval_train, time_interval_test, extra_inputs_train, extra_inputs_test = \
            interpolate_data(hidden_train, hidden_test, obs_train, obs_test, input_train, input_test,
                             extra_inputs_train, extra_inputs_test, FLAGS.use_gp, interpolation_data)

        if FLAGS.data_type in PERCENTAGE_DATA_DICT and FLAGS.emission == "mvn":
            # transform to log additive ratio
            percentage_train = []
            for i in range(len(obs_train)):
                percentage_train.append(obs_train[i])
                obs_train[i] = np.log(obs_train[i][:,:-1]) - np.log(obs_train[i][:, -1:])

            percentage_test = []
            for i in range(len(obs_test)):
                percentage_test.append(obs_test[i])
                obs_test[i] = np.log(obs_test[i][:, :-1]) - np.log(obs_test[i][:, -1:])
    else:
        raise ValueError("Data type must be one of available data types.")

    if FLAGS.use_mask:
        mask_train, mask_test = _mask_train, _mask_test
    else:
        # set all the elements of mask to be True
        mask_train = [np.ones_like(m, dtype=bool) for m in _mask_train]
        mask_test = [np.ones_like(m, dtype=bool) for m in _mask_test]

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
                         "epochs":         FLAGS.epochs,
                         "seed":          FLAGS.seed,
                         "rslt_dir_name": FLAGS.rslt_dir_name}

    RLT_DIR = create_RLT_DIR(Experiment_params)
    save_experiment_param(RLT_DIR, FLAGS)
    print("checkpoint_dir:", RLT_DIR)

    # ============================================= training part ============================================ #
    mytrainer = trainer(SSM_model, SMC_train, FLAGS)
    mytrainer.set_data_saving()
    mytrainer.init_train(obs_train, obs_test, hidden_train, hidden_test, input_train, input_test,
                         mask_train, mask_test, time_interval_train, time_interval_test,
                         extra_inputs_train, extra_inputs_test)

    plot_start_idx = 0
    for checkpoint_idx, epoch in enumerate(FLAGS.epochs):
        print("\n\nStart training {}...".format(checkpoint_idx))

        checkpoint_dir = RLT_DIR + "checkpoint_{}/".format(checkpoint_idx)
        print("Creating checkpoint_{} directory...".format(checkpoint_idx))
        os.makedirs(checkpoint_dir)

        mytrainer.set_saving_dir(checkpoint_dir)

        history, log = mytrainer.train(obs_train,
                                       hidden_train,
                                       input_train,
                                       mask_train,
                                       time_interval_train,
                                       extra_inputs_train,
                                       print_freq, epoch)

        # ======================================== data saving part ======================================== #

        with open(checkpoint_dir + "history.json", "w") as f:
            json.dump(history, f, indent=4, cls=NumpyEncoder)

        Xs, y_hat = log["Xs"], log["y_hat_original"]
        Xs_val = mytrainer.evaluate(Xs, mytrainer.test_feed_dict)

        y_hat_val_train = mytrainer.evaluate(y_hat, mytrainer.train_feed_dict)
        y_hat_val_test = mytrainer.evaluate(y_hat, mytrainer.test_feed_dict)
        print("Finish evaluating training results...")

        if FLAGS.emission == "mvn":
            # transform log additive ratio back to observation

            percentage_hat_val_train = [[[] for _ in range(FLAGS.n_train)] for _ in range(FLAGS.MSE_steps+1)]
            for i in range(len(y_hat_val_train)):
                # y hat val = (batch_size, n_days, Dy)
                for j in range(len(y_hat_val_train[i])):
                    n_days = y_hat_val_train[i][j].shape[0]  # (n_days, Dy)

                    percentage = np.concatenate((y_hat_val_train[i][j], np.zeros((n_days, 1))), axis=-1)  # (n_days, Dy+1)
                    percentage = \
                        np.exp(percentage - logsumexp(percentage, axis=-1, keepdims=True))
                    percentage_hat_val_train[i][j] = percentage

            percentage_hat_val_test = [[[] for _ in range(FLAGS.saving_test_num)] for _ in range(FLAGS.MSE_steps+1)]
            for i in range(len(y_hat_val_test)):
                # y hat val = (batch_size, n_days, Dy)
                for j in range(len(y_hat_val_test[i])):
                    n_days = y_hat_val_test[i][j].shape[0] # (n_days, Dy)

                    percentage = np.concatenate((y_hat_val_test[i][j], np.zeros((n_days, 1))), axis=-1)  # (n_days, Dy+1)
                    percentage = \
                        np.exp(percentage - logsumexp(percentage, axis=-1, keepdims=True))

                    percentage_hat_val_test[i][j] = percentage

            obs_train, obs_test, y_hat_val_train, y_hat_val_test = \
                percentage_train, percentage_test, percentage_hat_val_train, percentage_hat_val_test

        plot_y_hat(checkpoint_dir + "y_hat_train_plots", y_hat_val_train, obs_train, mask=mask_train,
                   saving_num=FLAGS.saving_train_num)
        plot_y_hat(checkpoint_dir + "y_hat_test_plots", y_hat_val_test, obs_test, mask=mask_test, saving_num=FLAGS.saving_test_num)
        plot_y_hat_bar_plot(checkpoint_dir+"train_obs_y_hat_bar_plots", y_hat_val_train, obs_train, mask=mask_train,
                            saving_num=FLAGS.saving_train_num, to_normalize=y_hat_bar_plot_to_normalize)
        plot_y_hat_bar_plot(checkpoint_dir+"test_obs_y_hat_bar_plots", y_hat_val_test, obs_test, mask=mask_test,
                            saving_num=FLAGS.saving_test_num, to_normalize=y_hat_bar_plot_to_normalize)

        if Dx == 2:
            plot_fhn_results(checkpoint_dir, Xs_val)
        if Dx == 3:
            plot_lorenz_results(checkpoint_dir, Xs_val)

        testing_data_dict = {"hidden_test": hidden_test[0:FLAGS.saving_test_num],
                             "obs_test": obs_test[0:FLAGS.saving_test_num],
                             "input_test": input_test[0:FLAGS.saving_test_num]}

        learned_model_dict = {"Xs_val": Xs_val,
                              "y_hat_val_test": y_hat_val_test}
        data_dict = {"testing_data_dict": testing_data_dict,
                     "learned_model_dict": learned_model_dict}

        with open(checkpoint_dir + "data.p", "wb") as f:
            pickle.dump(data_dict, f)

        # plot_MSEs(checkpoint_dir, history["MSE_trains"], history["MSE_tests"], print_freq)
        #
        # plot_R_square(checkpoint_dir, history["R_square_trains"][plot_start_idx:],
        #              history["R_square_tests"][plot_start_idx:], plot_start_idx, print_freq)
        plot_log_ZSMC(checkpoint_dir, history["log_ZSMC_trains"][plot_start_idx:],
                      history["log_ZSMC_tests"][plot_start_idx:], plot_start_idx, print_freq)
        plot_start_idx += int(epoch / print_freq) + 1

        y_hat_unmasked = mytrainer.evaluate(mytrainer.y_hat_unmasked_N_BxTxDy, mytrainer.train_all_feed_dict)
        y_hat_train_0_step = y_hat_unmasked[0]
        with open(os.path.join(checkpoint_dir, "y_hat_train.p"), "wb") as f:
            pickle.dump(y_hat_train_0_step, f)

        print("finish plotting!")

