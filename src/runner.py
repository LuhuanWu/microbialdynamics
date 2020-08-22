import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

# for data saving stuff

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

from src.utils.available_data import DATA_DIR_DICT, COUNT_DATA_DICT
from src.utils.data_loader import load_data
from src.utils.data_interpolation import interpolate_data

def main(_):
    FLAGS = tf.app.flags.FLAGS

    # ========================================= parameter part begins ========================================== #
    Dx = FLAGS.Dx
    print_freq = FLAGS.print_freq

    # evaluation parameters
    if FLAGS.g_dist_type in ["poisson", "multinomial", "multinomial_compose"]:
        y_hat_bar_plot_to_normalize = True
    else:
        raise ValueError("Unsupported emission!")

    FLAGS.epochs = [int(epoch) for epoch in FLAGS.epochs.split(",")]

    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # ============================================= dataset part ============================================= #
    # generate data from simulation
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_dir = DATA_DIR_DICT[FLAGS.data_type]
    data_dir = os.path.join(repo_dir, data_dir)
    if FLAGS.interpolation_type == 'none':
        FLAGS.interpolation_type = None

    hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, theta, params = \
        load_data(data_dir, train_num=FLAGS.train_num, test_num=FLAGS.test_num)

    FLAGS.n_train, FLAGS.n_test = n_train, n_test = len(obs_train), len(obs_test)

    hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, \
        mask_train, mask_test, time_interval_train, time_interval_test = \
        interpolate_data(hidden_train, hidden_test,
                         obs_train, obs_test,
                         input_train, input_test,
                         interpolation_type=FLAGS.interpolation_type,
                         pseudo_count=FLAGS.pseudo_count)

    # clip saving_test_num to avoid it > n_train or n_test
    min_time_train = min([obs.shape[0] for obs in obs_train])
    min_time_test = min([obs.shape[0] for obs in obs_test])
    min_time = min(min_time_train, min_time_test)
    FLAGS.MSE_steps = min(FLAGS.MSE_steps, min_time - 2)
    FLAGS.saving_train_num = min(FLAGS.saving_train_num, n_train)
    FLAGS.saving_test_num = min(FLAGS.saving_test_num, n_test)

    print("finished preparing dataset")

    # ============================================== model part ============================================== #
    SSM_model = SSM(FLAGS, theta)

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

    # =========================================== data saving part =========================================== #
    # create dir to save results
    Experiment_params = {"n_train":       n_train,
                         "n_test":        n_test,
                         "np":            FLAGS.n_particles,
                         "lr":            FLAGS.lr,
                         "epochs":        FLAGS.epochs,
                         "seed":          FLAGS.seed,
                         "rslt_dir_name": FLAGS.rslt_dir_name}

    RLT_DIR = create_RLT_DIR(Experiment_params)
    save_experiment_param(RLT_DIR, FLAGS)
    print("checkpoint_dir:", RLT_DIR)

    # ============================================= training part ============================================ #
    mytrainer = trainer(SSM_model, SMC_train, FLAGS)
    mytrainer.set_data_saving(RLT_DIR)
    mytrainer.init_train(obs_train, obs_test, input_train, input_test, mask_train, mask_test,
                         time_interval_train, time_interval_test)

    plot_start_idx, plot_start_epoch = 0, 0
    cum_epoch = 0
    for checkpoint_idx, epoch in enumerate(FLAGS.epochs):
        cum_epoch += epoch
        print("\n\nStart training {}...".format(checkpoint_idx))

        checkpoint_dir = RLT_DIR + "checkpoint_{}/".format(checkpoint_idx)
        print("Creating checkpoint_{} directory...".format(checkpoint_idx))
        os.makedirs(checkpoint_dir)

        mytrainer.set_saving_dir(checkpoint_dir)

        history, log = mytrainer.train(print_freq, epoch)

        # ======================================== data saving part ======================================== #

        with open(checkpoint_dir + "history.json", "w") as f:
            json.dump(history, f, indent=4, cls=NumpyEncoder)

        Xs, y_hat = log["Xs_resampled"], log["y_hat"]
        Xs_train, y_hat_train = mytrainer.evaluate([Xs, y_hat], mytrainer.train_feed_dict)
        Xs_test, y_hat_test = mytrainer.evaluate([Xs, y_hat], mytrainer.test_feed_dict)

        learned_model_dict = {"Xs_test": Xs_test, "y_hat_val_test": y_hat_test.copy()}

        plot_y_hat_bar_plot(checkpoint_dir + "y_hat_train_bar_plots", y_hat_train, obs_train, mask=mask_train,
                            saving_num=FLAGS.saving_train_num, to_normalize=y_hat_bar_plot_to_normalize)
        plot_y_hat_bar_plot(checkpoint_dir + "y_hat_test_bar_plots", y_hat_test, obs_test, mask=mask_test,
                            saving_num=FLAGS.saving_test_num, to_normalize=y_hat_bar_plot_to_normalize)

        testing_data_dict = {"hidden_test": hidden_test[0:FLAGS.saving_test_num],
                             "obs_test": obs_test[0:FLAGS.saving_test_num],
                             "input_test": input_test[0:FLAGS.saving_test_num]}

        if FLAGS.f_tran_type in ["ilr_clv", "ilr_clv_taxon"]:
            f_tran_params = mytrainer.sess.run(SSM_model.f_tran.params,
                                               {SSM_model.training: False,
                                                SSM_model.annealing_frac: cum_epoch / np.sum(FLAGS.epochs)})
            with open(data_dir, "rb") as f:
                data = pickle.load(f)
            plot_interaction_matrix(checkpoint_dir + "interaction", f_tran_params, data)
        else:
            f_tran_params = None

        data_dict = {"testing_data_dict": testing_data_dict,
                     "learned_model_dict": learned_model_dict,
                     "f_tran_params": f_tran_params}

        with open(checkpoint_dir + "data.p", "wb") as f:
            pickle.dump(data_dict, f)

        plot_log_ZSMC(checkpoint_dir, history["log_ZSMC_trains"][plot_start_idx:],
                      history["log_ZSMC_tests"][plot_start_idx:], plot_start_epoch, print_freq)
        plot_start_idx += int(epoch / print_freq) + 1
        plot_start_epoch += epoch

        y_hat_unmasked_train = mytrainer.evaluate(mytrainer.unmasked_y_hat_N_BxTxDy, mytrainer.train_all_feed_dict)
        y_hat_unmasked_test = mytrainer.evaluate(mytrainer.unmasked_y_hat_N_BxTxDy, mytrainer.test_all_feed_dict)
        y_hat_train_0_step, y_hat_test_0_step = y_hat_unmasked_train[0], y_hat_unmasked_test[0]
        with open(os.path.join(checkpoint_dir, "y_hat.p"), "wb") as f:
            pickle.dump({"train": y_hat_train_0_step, "test": y_hat_test_0_step}, f)

        print("finish plotting!")

