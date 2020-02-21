import numpy as np
from scipy.special import logsumexp, softmax

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
    if FLAGS.g_dist_type == "dirichlet":
        y_hat_bar_plot_to_normalize = False
    elif FLAGS.g_dist_type == "poisson" or FLAGS.g_dist_type == "multinomial":
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

    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # ============================================= dataset part ============================================= #
    # generate data from simulation
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # repo = git.Repo('.', search_parent_directories=True)
    # repo_dir = repo.working_tree_dir  # microbialdynamics

    data_dir = DATA_DIR_DICT[FLAGS.data_type]
    data_dir = os.path.join(repo_dir, data_dir)
    if FLAGS.interpolation_type == "clv":
        interpolation_data_dir = INTERPOLATION_DATA_DICT[FLAGS.interpolation_data_type]
        interpolation_data_dir = os.path.join(repo_dir, interpolation_data_dir)
        with open(interpolation_data_dir, "rb") as f:
            interpolation_data = pickle.load(f)
        # TODO: make sure that interpolation matches the shape
    elif FLAGS.interpolation_type == 'none':
        FLAGS.interpolation_type = None
        interpolation_data = None
    else:
        interpolation_data = None

    if FLAGS.data_type == "toy":
        print("Use toy data")
        hidden_train, hidden_test, obs_train, obs_test, input_train, input_test = joblib.load(data_dir)
        print("Finish loading the toy data.")

        train_num, test_num = len(obs_train), len(obs_test)
        T_train, T_test = obs_train[0].shape[0], obs_test[0].shape[0]

        mask_train, mask_test = np.ones((train_num, T_train), dtype=bool), np.ones((test_num, T_test), dtype=bool)
        time_interval_train, time_interval_test = np.zeros_like(mask_train), np.zeros_like(mask_test)
        extra_inputs_train, extra_inputs_test = np.zeros((train_num, T_train)), np.zeros((test_num, T_test))

    elif FLAGS.data_type in PERCENTAGE_DATA_DICT or FLAGS.data_type in COUNT_DATA_DICT:
        hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, \
        extra_inputs_train, extra_inputs_test = \
            load_data(data_dir, Dx, training_sample_idx=training_sample_idx, test_sample_idx=test_sample_idx)
        n_train, n_test = len(obs_train), len(obs_test)
        FLAGS.n_train, FLAGS.n_test = n_train, n_test

        hidden_train, hidden_test, obs_train, obs_test, input_train, input_test, \
        mask_train, mask_test, time_interval_train, time_interval_test, extra_inputs_train, extra_inputs_test = \
            interpolate_data(hidden_train, hidden_test, obs_train, obs_test, input_train, input_test,
                             extra_inputs_train, extra_inputs_test,
                             interpolation_type=FLAGS.interpolation_type, interpolation_data=interpolation_data,
                             pseudo_count=FLAGS.pseudo_count)
    else:
        raise ValueError("Data type must be one of available data types.")

    # clip saving_test_num to avoid it > n_train or n_test
    min_time_train = min([obs.shape[0] for obs in obs_train])
    min_time_test = min([obs.shape[0] for obs in obs_test])
    min_time = min(min_time_train, min_time_test)
    FLAGS.MSE_steps = min(FLAGS.MSE_steps, min_time - 2)
    FLAGS.saving_train_num = min(FLAGS.saving_train_num, n_train)
    FLAGS.saving_test_num = min(FLAGS.saving_test_num, n_test)

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
    mytrainer.set_data_saving()
    mytrainer.init_train(obs_train, obs_test, input_train, input_test, mask_train, mask_test,
                         time_interval_train, time_interval_test, extra_inputs_train, extra_inputs_test)

    plot_start_idx = 0
    for checkpoint_idx, epoch in enumerate(FLAGS.epochs):
        print("\n\nStart training {}...".format(checkpoint_idx))

        checkpoint_dir = RLT_DIR + "checkpoint_{}/".format(checkpoint_idx)
        print("Creating checkpoint_{} directory...".format(checkpoint_idx))
        os.makedirs(checkpoint_dir)

        mytrainer.set_saving_dir(checkpoint_dir)

        history, log = mytrainer.train(print_freq, epoch)

        # ======================================== data saving part ======================================== #

        with open(checkpoint_dir + "history.json", "w") as f:
            json.dump(history, f, indent=4, cls=NumpyEncoder)

        Xs, y_hat = log["Xs"], log["y_hat_original"]
        Xs_val_train = mytrainer.evaluate(Xs, mytrainer.train_feed_dict)
        Xs_val_test = mytrainer.evaluate(Xs, mytrainer.test_feed_dict)

        y_hat_val_train = mytrainer.evaluate(y_hat, mytrainer.train_feed_dict)
        y_hat_val_test = mytrainer.evaluate(y_hat, mytrainer.test_feed_dict)
        print("Finish evaluating training results...")

        plot_y_hat(checkpoint_dir + "y_hat_train_plots", y_hat_val_train, obs_train, mask=mask_train,
                   saving_num=FLAGS.saving_train_num)
        plot_y_hat(checkpoint_dir + "y_hat_test_plots", y_hat_val_test, obs_test, mask=mask_test, saving_num=FLAGS.saving_test_num)
        plot_y_hat_bar_plot(checkpoint_dir+"train_obs_y_hat_bar_plots", y_hat_val_train, obs_train, mask=mask_train,
                            saving_num=FLAGS.saving_train_num, to_normalize=y_hat_bar_plot_to_normalize)
        plot_y_hat_bar_plot(checkpoint_dir+"test_obs_y_hat_bar_plots", y_hat_val_test, obs_test, mask=mask_test,
                            saving_num=FLAGS.saving_test_num, to_normalize=y_hat_bar_plot_to_normalize)

        if Dx == 2:
            plot_fhn_results(checkpoint_dir, Xs_val_test)
        if Dx == 3:
            plot_lorenz_results(checkpoint_dir, Xs_val_test)
        testing_data_dict = {"hidden_test": hidden_test[0:FLAGS.saving_test_num],
                             "obs_test": obs_test[0:FLAGS.saving_test_num],
                             "input_test": input_test[0:FLAGS.saving_test_num]}

        learned_model_dict = {"Xs_val_test": Xs_val_test,
                              "y_hat_val_test": y_hat_val_test}
        if FLAGS.g_tran_type == "LDA":
            if FLAGS.beta_constant:
                beta_val = mytrainer.sess.run(SSM_model.g_tran.beta, {SSM_model.training: False})
                learned_model_dict["topic"] = beta_val
                plot_x_bar_plot(checkpoint_dir + "x_train_bar_plots", Xs_val_train)
                plot_x_bar_plot(checkpoint_dir + "x_test_bar_plots", Xs_val_test)
                plot_topic_bar_plot(checkpoint_dir, beta_val)
            else:
                # batch_size, (time, n_particles, Dx+1, Dy-1)
                beta_logs_train = mytrainer.evaluate(log["beta_logs"], feed_dict_w_batches=mytrainer.train_feed_dict)
                beta_logs_val = mytrainer.evaluate(log["beta_logs"], feed_dict_w_batches=mytrainer.test_feed_dict)
                beta_train = []
                saving_num = 10
                for i in range(saving_num):
                    beta_log = beta_logs_train[i]
                    beta_log = np.mean(beta_log, axis=1)  # (time, Dx+1, Dy-1)
                    beta_log = np.concatenate([beta_log, np.zeros_like(beta_log[..., 0:1])], axis=-1)
                    beta = softmax(beta_log, axis=-1) # (time, Dx+1, Dy)
                    beta_train.append(beta)
                    plot_topic_bar_plot_across_time(checkpoint_dir+"topic_contents", beta, name="train_{}".format(i))
                beta_test = []
                for i in range(saving_num):
                    beta_log = beta_logs_val[i]
                    beta_log = np.mean(beta_log, axis=1)  # (time, Dx+1, Dy-1)
                    beta_log = np.concatenate([beta_log, np.zeros_like(beta_log[..., 0:1])], axis=-1)
                    beta = softmax(beta_log, axis=-1)  # (time, Dx+1, Dy)
                    beta_test.append(beta)
                    plot_topic_bar_plot_across_time(checkpoint_dir + "topic_contents", beta, name="test_{}".format(i))

                A_beta, g_beta, Wg_beta = mytrainer.sess.run([SSM_model.f_beta_tran.A_beta,
                                                              SSM_model.f_beta_tran.g_beta,
                                                              SSM_model.f_beta_tran.Wg_beta],
                                                             {SSM_model.training: False})
                betas = {"beta_train": beta_train, "beta_test": beta_test,
                         "A_beta": A_beta, "g_beta": g_beta, "Wg_beta": Wg_beta}

                with open(checkpoint_dir + "beta.p", "wb") as f:
                    pickle.dump(betas, f)
        data_dict = {"testing_data_dict": testing_data_dict,
                     "learned_model_dict": learned_model_dict}

        with open(checkpoint_dir + "data.p", "wb") as f:
            pickle.dump(data_dict, f)

        plot_log_ZSMC(checkpoint_dir, history["log_ZSMC_trains"][plot_start_idx:],
                      history["log_ZSMC_tests"][plot_start_idx:], plot_start_idx, print_freq)
        plot_start_idx += int(epoch / print_freq) + 1

        y_hat_unmasked_train = mytrainer.evaluate(mytrainer.unmasked_y_hat_N_BxTxDy, mytrainer.train_all_feed_dict)
        y_hat_unmasked_test = mytrainer.evaluate(mytrainer.unmasked_y_hat_N_BxTxDy, mytrainer.test_all_feed_dict)
        y_hat_train_0_step, y_hat_test_0_step = y_hat_unmasked_train[0], y_hat_unmasked_test[0]
        with open(os.path.join(checkpoint_dir, "y_hat.p"), "wb") as f:
            pickle.dump({"train": y_hat_train_0_step, "test": y_hat_test_0_step}, f)

        print("finish plotting!")

