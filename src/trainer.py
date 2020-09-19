import numpy as np
from sklearn.utils import shuffle
from scipy.special import softmax
import math

import tensorflow as tf
import os
import pickle
import time

import matplotlib.pyplot as plt


EPS = 1e-6


class StopTraining(Exception):
    pass


class trainer:
    def __init__(self, model, SMC, FLAGS):
        self.model = model
        self.SMC = SMC
        self.FLAGS = FLAGS

        self.Dx = self.FLAGS.Dx
        self.Dy = self.FLAGS.Dy
        self.Dv = self.FLAGS.Dv
        self.n_particles = self.FLAGS.n_particles

        self.use_mask = self.FLAGS.use_mask
        self.epochs = self.FLAGS.epochs

        self.MSE_steps = self.FLAGS.MSE_steps

        # useful for simulating training dynamics
        self.save_res = False

        self.init_placeholder()
        self.init_training_param()

    def init_placeholder(self):
        self.obs = self.model.obs
        self.input = self.model.input
        self.time = self.model.time
        self.mask = self.model.mask
        self.mask_weight = self.model.mask_weight
        self.time_interval = self.model.time_interval
        self.training = self.model.training
        self.annealing_frac = self.model.annealing_frac

    def init_training_param(self):
        self.batch_size = self.FLAGS.batch_size
        self.lr = self.FLAGS.lr

        # early stopping
        self.early_stop_patience = self.FLAGS.early_stop_patience
        self.bestCost = 0
        self.early_stop_count = 0

        # lr auto decreasing
        self.lr_reduce_factor = self.FLAGS.lr_reduce_factor
        self.lr_reduce_patience = self.FLAGS.lr_reduce_patience
        self.min_lr = self.FLAGS.min_lr
        self.lr_reduce_count = 0

    def set_data_saving(self, RLT_DIR):
        self.save_res = True
        self.save_trajectory = self.FLAGS.save_trajectory
        self.save_y_hat_train = self.FLAGS.save_y_hat_train
        self.saving_train_num = self.FLAGS.saving_train_num

        self.save_y_hat_test = self.FLAGS.save_y_hat_test
        self.saving_test_num = self.FLAGS.saving_test_num

        # metrics
        self.log_ZSMC_trains = []
        self.log_ZSMC_tests = []

        self.R_square_original_trains = []
        self.R_square_original_tests = []

        self.R_square_percentage_trains = []
        self.R_square_percentage_tests = []

        self.R_square_logp_trains = []
        self.R_square_logp_tests = []

        # tensorboard and model saver
        self.save_tensorboard = self.FLAGS.save_tensorboard
        self.save_model = self.FLAGS.save_model

        if self.save_model:
            self.saver = tf.train.Saver(max_to_keep=1)

        if self.save_tensorboard:
            self.writer = tf.summary.FileWriter(RLT_DIR)

    def set_saving_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

        # epoch data (trajectory, y_hat and quiver lattice)
        epoch_data_DIR = self.checkpoint_dir.split("/")
        epoch_data_DIR.insert(epoch_data_DIR.index("rslts") + 1, "epoch_data")
        self.epoch_data_DIR = "/".join(epoch_data_DIR)

    def init_train(self, obs_train, obs_test, input_train, input_test,
                   mask_train, mask_test, time_interval_train, time_interval_test):

        self.obs_train, self.obs_test = obs_train, obs_test
        self.input_train, self.input_test = input_train, input_test
        self.mask_train, self.mask_test = mask_train, mask_test
        self.time_interval_train, self.time_interval_test = time_interval_train, time_interval_test

        # define objective
        self.log_ZSMC, self.log = self.SMC.get_log_ZSMC(self.obs, self.input, self.time,
                                                        self.mask, self.time_interval, self.mask_weight)

        self.Xs = self.log["Xs_resampled"]
        self.particles = [self.Xs]
        self.y_hat_N_BxTxDy, self.y_N_BxTxDy, self.unmasked_y_hat_N_BxTxDy = \
            self.SMC.n_step_MSE(self.MSE_steps, self.particles, self.obs, self.input, self.mask)
        self.log["y_hat"] = self.y_hat_N_BxTxDy

        # set up feed_dict
        self.set_feed_dict()

        # regularization
        if self.model.f_tran_type in ["ilr_clv", "ilr_clv_taxon"]:
            reg = self.model.f_tran.regularization_loss()

        with tf.variable_scope("loss"):
            loss = -self.log_ZSMC
            tf.summary.scalar('neg_log_ZSMC', -self.log_ZSMC)

            if self.model.f_tran_type in ["ilr_clv", "ilr_clv_taxon"]:
                tf.summary.scalar('reg', reg)
                loss += reg

        with tf.variable_scope("train"):
            self.lr_holder = tf.placeholder(tf.float32, name="lr")
            optimizer = tf.train.AdamOptimizer(self.lr_holder)
            self.train_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()

        print("initializing variables...")
        self.sess.run(init)
        self.total_epoch_count = 0
        self.summary = tf.summary.merge_all()

    def set_feed_dict(self):
        # data up to saving_num
        self.train_feed_dict = {self.obs: self.obs_train[0:self.saving_train_num],
                                self.input: self.input_train[0: self.saving_train_num],
                                self.time: [obs.shape[0] for obs in self.obs_train[0: self.saving_train_num]],
                                self.mask: self.mask_train[0: self.saving_train_num],
                                self.time_interval: self.time_interval_train[0:self.saving_train_num],
                                self.training: [False] * self.saving_train_num}

        self.test_feed_dict = {self.obs: self.obs_test[0:self.saving_test_num],
                               self.input: self.input_test[0:self.saving_test_num],
                               self.time: [obs.shape[0] for obs in self.obs_test[0: self.saving_test_num]],
                               self.mask: self.mask_test[0:self.saving_test_num],
                               self.time_interval: self.time_interval_test[0:self.saving_test_num],
                               self.training: [False] * self.saving_test_num}
        # all data
        self.train_all_feed_dict = {self.obs: self.obs_train,
                                    self.input: self.input_train,
                                    self.time: [obs.shape[0] for obs in self.obs_train],
                                    self.mask: self.mask_train,
                                    self.time_interval: self.time_interval_train,
                                    self.training: [False] * len(self.obs_train)}

        self.test_all_feed_dict = {self.obs: self.obs_test,
                                   self.input: self.input_test,
                                   self.time: [obs.shape[0] for obs in self.obs_test],
                                   self.mask: self.mask_test,
                                   self.time_interval: self.time_interval_test,
                                   self.training: [False] * len(self.obs_test)}

    def train(self, print_freq, epoch):
        if self.save_res and self.save_tensorboard:
            self.writer.add_graph(self.sess.graph)

        def set_feed_dict(ph, val):
            self.train_feed_dict[ph] = [val] * self.saving_train_num
            self.test_feed_dict[ph] = [val] * self.saving_test_num
            self.train_all_feed_dict[ph] = [val] * len(self.obs_train)
            self.test_all_feed_dict[ph] = [val] * len(self.obs_test)

        for i in range(epoch):
            if self.use_mask:
                mask_weight = 0
            else:
                mask_weight = 1
            annealing_frac = (self.total_epoch_count + 1.0) / np.sum(self.epochs)

            set_feed_dict(self.mask_weight, mask_weight)
            set_feed_dict(self.annealing_frac, annealing_frac)
            start = time.time()

            if i == 0:
                self.evaluate_and_save_metrics(self.total_epoch_count)

            # training
            obs_train, input_train, mask_train, time_interval_train = \
                shuffle(self.obs_train, self.input_train, self.mask_train, self.time_interval_train)
            for j in range(0, len(obs_train), self.batch_size):
                assert self.batch_size == 1

                summary, _ = self.sess.run([self.summary, self.train_op],
                                           feed_dict={self.obs:            obs_train[j:j + self.batch_size],
                                                      self.input:          input_train[j:j + self.batch_size],
                                                      self.time:           obs_train[j].shape[0],
                                                      self.mask:           mask_train[j:j + self.batch_size],
                                                      self.mask_weight:    mask_weight,
                                                      self.time_interval:  time_interval_train[j:j + self.batch_size],
                                                      self.lr_holder:      self.lr,
                                                      self.training:       True,
                                                      self.annealing_frac: annealing_frac})
                if self.save_tensorboard:
                    self.writer.add_summary(summary, self.total_epoch_count * len(obs_train) + j)

            if (self.total_epoch_count + 1) % print_freq == 0:
                try:
                    self.evaluate_and_save_metrics(self.total_epoch_count)
                    self.adjust_lr(i, print_freq)
                except StopTraining:
                    break

                if self.save_res:
                    if self.save_trajectory:
                        Xs_val = self.evaluate(self.Xs, self.test_feed_dict, average=False)

                    if self.save_trajectory:
                        trajectory_dict = {"Xs": Xs_val}
                        with open(self.epoch_data_DIR + "trajectory_{}.p".format(i + 1), "wb") as f:
                            pickle.dump(trajectory_dict, f)

                    if self.save_y_hat_train:
                        y_hat_val_train = self.evaluate(self.y_hat_N_BxTxDy, self.train_feed_dict, average=False)[0]
                        y_hat_train_dict = {"y_hat_train": y_hat_val_train}
                        with open(self.epoch_data_DIR + "y_hat_train_{}.p".format(i + 1), "wb") as f:
                            pickle.dump(y_hat_train_dict, f)

                    if self.save_y_hat_test:
                        y_hat_val_test = self.evaluate(self.y_hat_N_BxTxDy, self.test_feed_dict, average=False)[0]
                        y_hat_test_dict = {"y_hat_test": y_hat_val_test}
                        with open(self.epoch_data_DIR + "y_hat_test_{}.p".format(i + 1), "wb") as f:
                            pickle.dump(y_hat_test_dict, f)

            self.total_epoch_count += 1
            if self.FLAGS.print_f_sigma:
                print("f sig", self.sess.run(self.model.f_dist.sigma_con))
            end = time.time()
            print("epoch {:<14} took {:.3f} seconds".format(self.total_epoch_count, end - start))
        print("finished training...")

        metrics = {"log_ZSMC_trains":            self.log_ZSMC_trains,
                   "log_ZSMC_tests":             self.log_ZSMC_tests,
                   "R_square_trains":            self.R_square_original_trains,
                   "R_square_tests":             self.R_square_original_tests,
                   "R_square_percentage_trains": self.R_square_percentage_trains,
                   "R_square_percentage_tests":  self.R_square_percentage_tests,
                   "R_square_logp_trains":       self.R_square_logp_trains,
                   "R_square_logp_tests":        self.R_square_logp_tests}

        return metrics, self.log

    def close_session(self):
        self.sess.close()

    def evaluate_and_save_metrics(self, iter_num):

        [log_ZSMC_train, y_hat_train, y_train] = \
            self.evaluate([self.log_ZSMC, self.y_hat_N_BxTxDy, self.y_N_BxTxDy],
                          feed_dict_w_batches=self.train_all_feed_dict)

        [log_ZSMC_test, y_hat_test, y_test] = \
            self.evaluate([self.log_ZSMC, self.y_hat_N_BxTxDy, self.y_N_BxTxDy],
                          feed_dict_w_batches=self.test_all_feed_dict)

        log_ZSMC_train, log_ZSMC_test = np.mean(log_ZSMC_train), np.mean(log_ZSMC_test)

        R_square_original_train, R_square_percentage_train, R_square_logp_train = \
            self.evaluate_R_square(y_hat_train, y_train)
        R_square_original_test, R_square_percentage_test, R_square_logp_test = \
            self.evaluate_R_square(y_hat_test, y_test)

        print()
        print("iter", iter_num + 1)
        print("Train log_ZSMC: {:>7.3f}, valid log_ZSMC: {:>7.3f}".format(log_ZSMC_train, log_ZSMC_test))

        print("Train, Valid k-step Rsq (original space):\n", R_square_original_train, "\n", R_square_original_test)
        print("Train, Valid k-step Rsq (percent space):\n", R_square_percentage_train, "\n", R_square_percentage_test)

        if not math.isfinite(log_ZSMC_train):
            print("Nan in log_ZSMC, stop training")
            raise StopTraining()

        if self.save_res:
            self.log_ZSMC_trains.append(log_ZSMC_train)
            self.log_ZSMC_tests.append(log_ZSMC_test)

            self.R_square_original_trains.append(R_square_original_train)
            self.R_square_original_tests.append(R_square_original_test)

            self.R_square_percentage_trains.append(R_square_percentage_train)
            self.R_square_percentage_tests.append(R_square_percentage_test)

            self.R_square_logp_trains.append(R_square_logp_train)
            self.R_square_logp_tests.append(R_square_logp_test)

            if not os.path.exists(self.epoch_data_DIR):
                os.makedirs(self.epoch_data_DIR)
            metric_dict = {"log_ZSMC_train": log_ZSMC_train,
                           "log_ZSMC_test":  log_ZSMC_test,
                           "R_square_percentage_train": R_square_percentage_train,
                           "R_square_percentage_test":  R_square_percentage_test}
            with open(self.epoch_data_DIR + "metric_{}.p".format(iter_num + 1), "wb") as f:
                pickle.dump(metric_dict, f)

    def adjust_lr(self, iter_num, print_freq):
        # determine whether should decrease lr or even stop training
        if self.FLAGS.data_type in ["count", "percentage"]:
            best_cost = np.argmax(self.log_ZSMC_trains)
        else:
            best_cost = np.argmax(self.log_ZSMC_tests)
        if self.bestCost != best_cost:
            self.early_stop_count = 0
            self.lr_reduce_count = 0
            self.bestCost = best_cost

        print("best valid cost on iter: {}\n".format(self.bestCost * print_freq))

        if self.bestCost != len(self.log_ZSMC_tests) - 1:
            self.early_stop_count += 1

            if self.early_stop_count * print_freq == self.early_stop_patience:
                print("valid cost not improving. stopping training...")
                raise StopTraining()

            self.lr_reduce_count += 1
            if self.lr_reduce_count * print_freq >= self.lr_reduce_patience:
                self.lr_reduce_count = 0
                self.lr = max(self.lr * self.lr_reduce_factor, self.min_lr)
                print("valid cost not improving. reduce learning rate to {}".format(self.lr))

        if self.save_model:
            if not os.path.exists(self.checkpoint_dir + "model/"):
                os.makedirs(self.checkpoint_dir + "model/")
            if self.bestCost == len(self.log_ZSMC_tests) - 1:
                print("Test log_ZSMC improves to {}, save model".format(self.log_ZSMC_tests[-1]))
                self.saver.save(self.sess, self.checkpoint_dir + "model/model_epoch", global_step=iter_num + 1)

    def evaluate(self, fetches, feed_dict_w_batches={}, average=False, keepdims=False):
        """
        Evaluate fetches across multiple batches of feed_dict
        fetches: a single tensor or list of tensor to evaluate
        feed_dict_w_batches: {placeholder: input of multiple batches}
        average: whether to average fetched values across batches
        keepdims: if not averaging across batches, for N-d tensor in feteches, whether to keep
            the dimension for different batches.
        """
        if not feed_dict_w_batches:
            return self.sess.run(fetches)

        n_batches = len(list(feed_dict_w_batches.values())[0])
        assert n_batches >= self.batch_size

        fetches_list = []
        feed_dict = {}
        for i in range(0, n_batches, self.batch_size):
            for key, value in feed_dict_w_batches.items():
                value = value[i:i + self.batch_size]
                if len(value) == 1 and not hasattr(value[0], "__len__"):
                    value = value[0]
                feed_dict[key] = value
            fetches_val = self.sess.run(fetches, feed_dict=feed_dict)
            if fetches == self.log_ZSMC:
                if not math.isfinite(fetches_val):
                    print(i, fetches_val)
            fetches_list.append(fetches_val)

        res = []
        if isinstance(fetches, list):
            for i in range(len(fetches)):
                if isinstance(fetches_list[0][i], np.ndarray):
                    all_array = [x[i] for x in fetches_list]
                    if not all(x.shape == all_array[0].shape for x in all_array):
                        tmp = [x[0] for x in all_array]
                    elif keepdims:
                        tmp = np.stack(all_array)
                    else:
                        tmp = np.concatenate(all_array)
                elif isinstance(fetches_list[0][i], list):
                    # should be y_hat_N_BxTmkxDy and y_N_BxTmkxDy
                    tmp = [[x[i][j] for x in fetches_list] for j in range(len(fetches[i]))]
                else:
                    tmp = np.array([x[i] for x in fetches_list])
                res.append(tmp)
        else:
            if isinstance(fetches_list[0], np.ndarray):
                if not all(x.shape == fetches_list[0].shape for x in fetches_list):
                    assert fetches_list[0].shape[0] == 1
                    res = [x[0] for x in fetches_list]
                else:
                    res = np.stack(fetches_list) if keepdims else np.concatenate(fetches_list)
            else:
                res = np.array(fetches_list)

        if average:
            if isinstance(res, list):
                res = [np.mean(x, axis=0) for x in res]
            else:
                res = np.mean(res, axis=0)

        return res

    @staticmethod
    def evaluate_R_square(y_hat_N_BxTxDy, y_N_BxTxDy):

        def R_square(y_hat_i, y_i):
            y_hat_i = np.concatenate(y_hat_i, axis=0)
            y_i = np.concatenate(y_i, axis=0)
            RMSE = np.sqrt(np.sum((y_hat_i - y_i) ** 2) / y_hat_i.shape[0])
            return RMSE
            # Rsq = []
            # for y_hat_i_j, y_i_j in zip(y_hat_i, y_i):
            #     num_timestamps = y_i_j.shape[0]
            #     MSE = np.sum((y_hat_i_j - y_i_j) ** 2)
            #     y_i_j_mean = np.mean(y_i_j, axis=0, keepdims=True)
            #     y_i_j_var = np.sum((y_i_j - y_i_j_mean) ** 2)
            #     Rsq.extend([1 - MSE / y_i_j_var] * num_timestamps)
            #     if not math.isfinite(1 - MSE / y_i_j_var):
            #         print(MSE, y_i_j_var)
            # return np.mean(Rsq)

        n_steps = len(y_hat_N_BxTxDy) - 1
        R_square_count = np.zeros(n_steps + 1)
        R_square_percentage = np.zeros(n_steps + 1)
        R_square_logp = np.zeros(n_steps + 1)
        for i, (y_hat_i, y_i) in enumerate(zip(y_hat_N_BxTxDy, y_N_BxTxDy)):
            # remove redundant dimension
            y_hat_i = [ele[0] for ele in y_hat_i]  # list of (T, Dy) arrays
            y_i = [ele[0] for ele in y_i]

            Dy = y_i[0].shape[1]

            p_hat_i = [y_hat_i_j / np.sum(y_hat_i_j, axis=-1, keepdims=True) for y_hat_i_j in y_hat_i]
            p_i = [y_i_j / np.sum(y_i_j, axis=-1, keepdims=True) for y_i_j in y_i]

            logp_hat_i = [np.log((p_hat_i_j + 1e-6) / (1 + 1e-6 * Dy)) for p_hat_i_j in p_hat_i]
            logp_i = [np.log((p_i_j + 1e-6) / (1 + 1e-6 * Dy)) for p_i_j in p_i]

            R_square_count[i] = R_square(y_hat_i, y_i)
            R_square_percentage[i] = R_square(p_hat_i, p_i)
            R_square_logp[i] = R_square(logp_hat_i, logp_i)

        return R_square_count, R_square_percentage, R_square_logp

    def draw_2D_quiver_plot(self, Xs_val, epoch):
        # Xs_val.shape = (saving_test_num, time, n_particles, Dx)

        plt.figure()
        for X_traj in Xs_val[0:self.saving_test_num]:
            X_traj = np.mean(X_traj, axis=1)
            plt.plot(X_traj[:, 0], X_traj[:, 1])
            plt.scatter(X_traj[0, 0], X_traj[0, 1])
        plt.title("quiver")
        plt.xlabel("x_dim 1")
        plt.ylabel("x_dim 2")

        # sns.despine()
        if not os.path.exists(self.checkpoint_dir + "quiver/"):
            os.makedirs(self.checkpoint_dir + "quiver/")
        plt.savefig(self.checkpoint_dir + "quiver/epoch_{}".format(epoch))
        plt.close()

    def define2Dlattice(self, x1range=(-30.0, 30.0), x2range=(-30.0, 30.)):

        x1coords = np.linspace(x1range[0], x1range[1], num=self.lattice_shape[0])
        x2coords = np.linspace(x2range[0], x2range[1], num=self.lattice_shape[1])
        Xlattice = np.stack(np.meshgrid(x1coords, x2coords), axis=-1)
        return Xlattice

    def draw_3D_quiver_plot(self, Xs_val, epoch):
        # Xs_val.shape = (saving_test_num, time, n_particles, Dx)

        fig = plt.figure()
        ax = fig.gca(projection="3d")
        plt.title("hidden state for all particles")
        ax.set_xlabel("x_dim 1")
        ax.set_ylabel("x_dim 2")
        ax.set_zlabel("x_dim 3")
        for X_traj in Xs_val[0:self.saving_test_num]:
            X_traj = np.mean(X_traj, axis=1)
            ax.plot(X_traj[:, 0], X_traj[:, 1], X_traj[:, 2])
            ax.scatter(X_traj[0, 0], X_traj[0, 1], X_traj[0, 2])

        if not os.path.exists(self.checkpoint_dir + "quiver/"):
            os.makedirs(self.checkpoint_dir + "quiver/")
        for angle in range(45, 360, 45):
            ax.view_init(30, angle)
            plt.savefig(self.checkpoint_dir + "quiver/epoch_{}_angle_{}".format(epoch, angle))
        plt.close()
