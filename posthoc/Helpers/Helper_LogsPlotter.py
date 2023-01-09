import matplotlib.pyplot as plt
import numpy as np

class LogsPlotter():
    def __init__(self, config, logs):
        self.logs = logs
        self.config = config

    def sleep_plot_losses_old(self):
        train_loss = np.array([self.logs["train_logs"][i]["train_loss"] for i in self.logs["train_logs"]])

        # for i in range(len(train_loss)):
        #     if i%100 == 0: print("{}_{}".format(i, train_loss[i]))

        val_loss = np.array([self.logs["val_logs"][i]["val_loss"] for i in self.logs["val_logs"]])
        steps = np.array([i / self.logs["train_logs"][i]["validate_every"] for i in self.logs["train_logs"]]) - 1

        plt.figure()
        plt.plot(steps, train_loss, label="Train")
        plt.plot(steps, val_loss, label="Valid")

        best_step = self.logs["best_logs"]["step"] / self.logs["train_logs"][self.logs["best_logs"]["step"]]["validate_every"] - 1
        best_loss = self.logs["best_logs"]["val_loss"]

        plt.plot((best_step, best_step), (0, best_loss), linestyle="--", color="y", label="Chosen Point")
        plt.plot((0, best_step), (best_loss, best_loss), linestyle="--", color="y")

        if self.config.rec_test:
            test_loss = np.array([self.logs["test_logs"][i]["test_loss"] for i in self.logs["test_logs"]])
            steps = np.array([i / self.logs["train_logs"][i]["validate_every"] for i in self.logs["test_logs"]]) - 1
            best_test_step = np.argmin(test_loss)
            best_test_loss = test_loss[best_test_step]
            plt.plot(steps, test_loss, label="Test")
            plt.plot((best_test_step, best_test_step), (0, best_test_loss), linestyle="--", color="r",
                     label="Chosen Point")
            plt.plot((0, best_test_step), (best_test_loss, best_test_loss), linestyle="--", color="r")

        plt.xlabel('Steps')
        plt.ylabel('Loss Values')
        plt.title("Loss")
        loss_min = np.min([np.min(train_loss), np.min(val_loss)])
        loss_max = np.max([np.max(train_loss), np.max(val_loss)])
        plt.ylim([loss_min - 0.05, loss_max + 0.05])
        plt.legend()
        # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/loss.png")
        plt.show()

    def sleep_plot_losses(self, chosen_loss = None):

        list_losses = ["total"]
        if "multi_loss" in self.config.model.args:
            list_losses += ["ce_loss_{}".format(i) for i, v in self.config.model.args.multi_loss.multi_loss_weights.multi_supervised_loss.items() if v != 0]
            list_losses += [i for i, v in self.config.model.args.multi_loss.multi_loss_weights.items() if v != 0 and type(v) == int]

        if type(self.logs["train_logs"][list(self.logs["train_logs"].keys())[0]]["train_loss"]) != dict: return self.sleep_plot_losses_old()

        train_loss = { loss_key: np.array([self.logs["train_logs"][i]["train_loss"][loss_key] for i in self.logs["train_logs"]]) for loss_key in list_losses}
        val_loss = {loss_key: np.array([self.logs["val_logs"][i]["val_loss"][loss_key] for i in self.logs["val_logs"]]) for loss_key in list_losses}

        if chosen_loss and chosen_loss not in list_losses: raise ValueError("chosen loss doesnt exist, choose from {}".format(list_losses[1:]))

        steps = np.array([i / self.logs["train_logs"][i]["validate_every"] for i in self.logs["train_logs"]]) - 1
        best_step = (self.logs["best_logs"]["step"] / self.logs["train_logs"][self.logs["best_logs"]["step"]]["validate_every"]) - 1

        plt.figure()
        loss_min = 100
        loss_max = 0
        plotted_losses = 0
        for loss_key in list_losses:
            if loss_key == "total" or (chosen_loss and chosen_loss!=loss_key): continue
            # plt.plot(steps, (train_loss[loss_key] - train_loss[loss_key].mean())/train_loss[loss_key].std(), label="Train_{}".format(loss_key))
            # plt.plot(steps, (val_loss[loss_key] - val_loss[loss_key].mean())/val_loss[loss_key].std() , label="Valid_{}".format(loss_key))
            # loss_min = np.minimum(loss_min, np.min((val_loss[loss_key] - val_loss[loss_key].mean()) / val_loss[loss_key].std()))
            # loss_min = np.minimum(loss_min, np.min((train_loss[loss_key] - train_loss[loss_key].mean()) / train_loss[loss_key].std()))
            # loss_max = np.maximum(loss_max, np.max((val_loss[loss_key] - val_loss[loss_key].mean()) / val_loss[loss_key].std()))
            # loss_max = np.maximum(loss_max, np.max((train_loss[loss_key] - train_loss[loss_key].mean()) / train_loss[loss_key].std()))
            plt.plot(steps, (train_loss[loss_key]), label="Train_{}".format(loss_key))
            plt.plot(steps, (val_loss[loss_key]), label="Valid_{}".format(loss_key))
            loss_min = np.minimum(loss_min, np.min(train_loss[loss_key]))
            loss_min = np.minimum(loss_min, np.min(val_loss[loss_key]))
            loss_max = np.maximum(loss_max, np.max(train_loss[loss_key]))
            loss_max = np.maximum(loss_max, np.max(val_loss[loss_key]))

            best_loss = {loss_key: self.logs["best_logs"]["val_loss"][loss_key]}
            plt.plot((best_step, best_step), (0, best_loss[loss_key]), linestyle="--", color="y")
            plt.plot((0, best_step), (best_loss[loss_key], best_loss[loss_key]), linestyle="--", color="y")
            plotted_losses += 1

        if plotted_losses>1:
            plt.xlabel('Steps')
            plt.ylabel('Loss Values')
            plt.title("Individual Losses")
            if not np.isnan(loss_min) and not np.isnan(loss_max):
                plt.ylim([loss_min - 0.05, loss_max + 0.05])
            plt.legend()
            # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/loss.png")
            plt.show()

        plt.figure()
        loss_min = 100
        loss_max = 0
        loss_key = "total"
        plt.plot(steps, train_loss[loss_key], label="Train_{}".format(loss_key))
        plt.plot(steps, val_loss[loss_key], label="Valid_{}".format(loss_key))
        loss_min = np.minimum(loss_min, np.min(train_loss[loss_key]))
        loss_min = np.minimum(loss_min, np.min(val_loss[loss_key]))
        loss_max = np.maximum(loss_max, np.max(train_loss[loss_key]))
        loss_max = np.maximum(loss_max, np.max(val_loss[loss_key]))
        best_loss = {loss_key: self.logs["best_logs"]["val_loss"][loss_key]}
        plt.plot((best_step, best_step), (0, best_loss[loss_key]), linestyle="--", color="y", label="Chosen Point")
        plt.plot((0, best_step), (best_loss[loss_key], best_loss[loss_key]), linestyle="--", color="y")

        plt.xlabel('Steps')
        plt.ylabel('Loss Values')
        plt.title("Total Loss")
        plt.ylim([loss_min - 0.05, loss_max + 0.05])
        plt.legend()
        # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/loss.png")
        plt.show()

    def sleep_plot_performance(self):

        if "multi_loss" in self.config.model.args:
            list_predictors = ["{}".format(i) for i, v in self.config.model.args.multi_loss.multi_loss_weights.multi_supervised_loss.items() if v != 0]
        else:
            list_predictors = ["combined"]

        if type(self.logs["train_logs"][list(self.logs["train_logs"].keys())[0]]["train_f1"]) != dict: return self.sleep_plot_performance_old()

        train_f1 = {pred_key: np.array([self.logs["train_logs"][i]["train_f1"][pred_key] for i in self.logs["train_logs"]]) for pred_key in list_predictors}
        val_f1 = {pred_key: np.array([self.logs["val_logs"][i]["val_f1"][pred_key] for i in self.logs["val_logs"]]) for pred_key in list_predictors}

        steps = np.array([i / self.logs["train_logs"][i]["validate_every"] for i in self.logs["train_logs"]]) - 1
        best_step = self.logs["best_logs"]["step"] / self.logs["train_logs"][self.logs["best_logs"]["step"]]["validate_every"]


        for loss_key in list_predictors:
            plt.figure()
            loss_min = 100
            loss_max = 0
            plt.plot(steps, train_f1[loss_key], label="Train_{}".format(loss_key))
            plt.plot(steps, val_f1[loss_key], label="Valid_{}".format(loss_key))
            loss_min = np.minimum(loss_min, np.min(train_f1[loss_key]))
            loss_min = np.minimum(loss_min, np.min(val_f1[loss_key]))
            loss_max = np.maximum(loss_max, np.max(train_f1[loss_key]))
            loss_max = np.maximum(loss_max, np.max(val_f1[loss_key]))
            best_loss = {loss_key: self.logs["best_logs"]["val_f1"][loss_key]}
            plt.plot((best_step, best_step), (0, best_loss[loss_key]), linestyle="--", color="y", label="Chosen Point")
            plt.plot((0, best_step), (best_loss[loss_key], best_loss[loss_key]), linestyle="--", color="y")
            plt.xlabel('Steps')
            plt.ylabel('F1 Value')
            plt.title("F1 {} Predictors".format(loss_key))
            plt.ylim([loss_min - 0.05, loss_max + 0.05])
            plt.legend()
            # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/loss.png")
            plt.show()

        train_k = {pred_key: np.array([self.logs["train_logs"][i]["train_k"][pred_key] for i in self.logs["train_logs"]]) for pred_key in list_predictors}
        val_k = {pred_key: np.array([self.logs["val_logs"][i]["val_k"][pred_key] for i in self.logs["val_logs"]]) for pred_key in list_predictors}

        for loss_key in list_predictors:
            plt.figure()
            loss_min = 100
            loss_max = 0
            plt.plot(steps, train_k[loss_key], label="Train_{}".format(loss_key))
            plt.plot(steps, val_k[loss_key], label="Valid_{}".format(loss_key))
            loss_min = np.minimum(loss_min, np.min(train_k[loss_key]))
            loss_min = np.minimum(loss_min, np.min(val_k[loss_key]))
            loss_max = np.maximum(loss_max, np.max(train_k[loss_key]))
            loss_max = np.maximum(loss_max, np.max(val_k[loss_key]))
            best_loss = {loss_key: self.logs["best_logs"]["val_k"][loss_key]}
            plt.plot((best_step, best_step), (0, best_loss[loss_key]), linestyle="--", color="y", label="Chosen Point")
            plt.plot((0, best_step), (best_loss[loss_key], best_loss[loss_key]), linestyle="--", color="y")
            plt.xlabel('Steps')
            plt.ylabel('K Value')
            plt.title("K {} Predictors".format(loss_key))
            plt.ylim([loss_min - 0.05, loss_max + 0.05])
            plt.legend()
            # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/loss.png")
            plt.show()

        train_f1_perclass = { pred_key: np.array([self.logs["train_logs"][i]["train_perclassf1"][pred_key] for i in self.logs["train_logs"]]) for pred_key in list_predictors}
        val_f1_perclass = { pred_key: np.array([self.logs["val_logs"][i]["val_perclassf1"][pred_key] for i in self.logs["val_logs"]]) for pred_key in list_predictors}

        for pred_key in list_predictors:

            plt.figure()
            score_min = 100
            score_max = 0
            colors = ["b", "k", "r"]
            color_dict = {v: colors[i] for i, v in enumerate(list_predictors)}
            color_dict = {"Training":"b", "Validation":"r"}
            for set in [{"score": train_f1_perclass, "label": "Training"},
                        {"score": val_f1_perclass, "label": "Validation"}]:

                plt.plot(steps, set["score"][pred_key][:, 0], color=color_dict[set["label"]], label="{}".format(set["label"]), linewidth=0.4)
                plt.plot(steps, set["score"][pred_key][:, 1], color=color_dict[set["label"]], linewidth=0.4)
                plt.plot(steps, set["score"][pred_key][:, 2], color=color_dict[set["label"]], linewidth=0.4)
                plt.plot(steps, set["score"][pred_key][:, 3], color=color_dict[set["label"]], linewidth=0.4)
                plt.plot(steps, set["score"][pred_key][:, 4], color=color_dict[set["label"]], linewidth=0.4)

                score_min = np.minimum(score_min, np.min(set["score"][pred_key]))
                score_max = np.maximum(score_max, np.max(set["score"][pred_key]))
                if set["label"] == "Validation":
                    for i in range(5):
                        best_loss = self.logs["best_logs"]["val_perclassf1"][pred_key][i]
                        plt.plot((best_step, best_step), (0, best_loss), linestyle="--", color="y", linewidth=0.6)
                        plt.plot((0, best_step), (best_loss, best_loss), linestyle="--", color="y", linewidth=0.6)
                else:
                    plt.plot((best_step, best_step), (0, score_max), linestyle="--", color="y", linewidth=0.6)

            plt.plot((0, steps[-1]), (0.8, 0.8), linestyle="--", linewidth=0.4, color="k")
            plt.plot((0, steps[-1]), (0.85, 0.85), linestyle="--", linewidth=0.4, color="k")
            plt.plot((0, steps[-1]), (0.9, 0.9), linestyle="--", linewidth=0.4, color="k")
            plt.plot((0, steps[-1]), (0.95, 0.95), linestyle="--", linewidth=0.4, color="k")

            plt.xlabel('Steps')
            plt.ylabel('F1 Value')
            plt.title("F1 Multi Predictors on {}".format(set["label"]))
            plt.yticks([0.4, 0.45, 0.5, 0.55, 0.8, 0.85, 0.9, 0.95])
            plt.ylim([score_min - 0.05, score_max + 0.05])
            plt.legend()
            # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/loss.png")
            plt.show()

    def sleep_plot_performance_old(self):

        train_f1 = np.array([self.logs["train_logs"][i]["train_f1"] for i in self.logs["train_logs"]])
        val_f1 = np.array([self.logs["val_logs"][i]["val_f1"] for i in self.logs["val_logs"]])
        steps = np.array([i / self.logs["train_logs"][i]["validate_every"] for i in self.logs["train_logs"]]) - 1

        plt.figure()
        plt.plot(steps, train_f1, label="Train")
        plt.plot(steps, val_f1, label="Valid")

        best_step = self.logs["best_logs"]["step"] / self.logs["train_logs"][self.logs["best_logs"]["step"]]["validate_every"] - 1
        best_f1 = self.logs["best_logs"]["val_f1"]

        plt.plot((best_step, best_step), (0, best_f1), linestyle="--", color="y", label="Chosen Point")
        plt.plot((0, best_step), (best_f1, best_f1), linestyle="--", color="y")

        if self.config.rec_test:
            test_f1 = np.array([self.logs["test_logs"][i]["test_f1"] for i in self.logs["test_logs"]])
            best_test_step = np.argmax(test_f1)
            best_test_f1 = test_f1[best_test_step]
            plt.plot(steps, test_f1, label="Test")
            plt.plot((best_test_step, best_test_step), (0, best_test_f1), linestyle="--", color="r",
                     label="Chosen Point")
            plt.plot((0, best_test_step), (best_test_f1, best_test_f1), linestyle="--", color="r")

        plt.xlabel('Steps')
        plt.ylabel('F1')
        plt.title("Validation F1 ")
        f1_min = np.min([np.min(train_f1), np.min(val_f1)])
        f1_max = np.max([np.max(train_f1), np.max(val_f1)])
        plt.ylim([f1_min - 0.05, f1_max + 0.05])
        plt.legend()
        # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/f1.png")
        plt.show()

        train_k = np.array([self.logs["train_logs"][i]["train_k"] for i in self.logs["train_logs"]])
        val_k = np.array([self.logs["val_logs"][i]["val_k"] for i in self.logs["val_logs"]])
        steps = np.array([i / self.logs["train_logs"][i]["validate_every"] for i in self.logs["train_logs"]]) - 1

        plt.figure()
        plt.plot(steps, train_k, label="Train")
        plt.plot(steps, val_k, label="Valid")

        best_step = self.logs["best_logs"]["step"] / self.logs["train_logs"][self.logs["best_logs"]["step"]]["validate_every"] - 1
        best_k = self.logs["best_logs"]["val_k"]

        plt.plot((best_step, best_step), (0, best_k), linestyle="--", color="y", label="Chosen Point")
        plt.plot((0, best_step), (best_k, best_k), linestyle="--", color="y")

        if self.config.rec_test:
            test_k = np.array([self.logs["test_logs"][i]["test_k"] for i in self.logs["test_logs"]])
            best_test_step = np.argmax(test_k)
            best_test_k = test_k[best_test_step]
            plt.plot(steps, test_k, label="Test")
            plt.plot((best_test_step, best_test_step), (0, best_test_k), linestyle="--", color="r",
                     label="Chosen Point")
            plt.plot((0, best_test_step), (best_test_k, best_test_k), linestyle="--", color="r")

        plt.xlabel('Steps')
        plt.ylabel('Kappa')
        plt.title("Cohen's kappa")
        plt.legend()
        kappa_min = np.min([np.min(train_k), np.min(val_k)])
        kappa_max = np.max([np.max(train_k), np.max(val_k)])
        plt.ylim([kappa_min, kappa_max + 0.05])
        # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/kappa.png")
        plt.show()

        val_f1 = np.array([self.logs["val_logs"][i]["val_perclassf1"] for i in self.logs["val_logs"]])
        steps = np.array([i / self.logs["train_logs"][i]["validate_every"] for i in self.logs["train_logs"]]) - 1

        plt.figure()
        plt.plot(steps, val_f1[:, 0], label="Wake", linewidth=0.8)
        plt.plot(steps, val_f1[:, 1], label="N1", linewidth=0.8)
        plt.plot(steps, val_f1[:, 2], label="N2", linewidth=0.8)
        plt.plot(steps, val_f1[:, 3], label="N3", linewidth=0.8)
        plt.plot(steps, val_f1[:, 4], label="REM", linewidth=0.8)

        f1_min = np.min(val_f1)
        f1_max = np.max(val_f1)
        best_step = self.logs["best_logs"]["step"] / self.logs["train_logs"][self.logs["best_logs"]["step"]]["validate_every"] - 1
        best_f1 = self.logs["best_logs"]["val_f1"]

        plt.plot((best_step, best_step), (0, f1_max + 0.05), linestyle="--", color="y", label="Chosen Point")
        plt.plot((0, steps[-1]), (0.8, 0.8), linestyle="--", linewidth=0.4, color="k")
        plt.plot((0, steps[-1]), (0.85, 0.85), linestyle="--", linewidth=0.4, color="k")
        plt.plot((0, steps[-1]), (0.9, 0.9), linestyle="--", linewidth=0.4, color="k")
        plt.plot((0, steps[-1]), (0.95, 0.95), linestyle="--", linewidth=0.4, color="k")
        # plt.plot((0, best_step), (best_f1, best_f1), linestyle="--", color="b")

        plt.xlabel('Steps')
        plt.ylabel('F1')
        plt.title("Validation F1 per class")

        plt.ylim([f1_min - 0.05, 1.01])
        plt.legend()
        # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/f1_perclass.png")
        plt.show()

    def sleep_plot_lr(self):

        learning_rate = np.concatenate([np.array(self.logs["train_logs"][i]["learning_rate"]) for i in self.logs["train_logs"]]).flatten()
        validation_steps = np.arange(0, len(learning_rate))
        checkpoint_steps = np.array([i / self.logs["train_logs"][i]["validate_every"] for i in self.logs["train_logs"]]) - 1

        plt.figure()
        plt.plot(validation_steps, learning_rate)
        plt.xticks(validation_steps[0:-1:self.logs["train_logs"][list(self.logs["train_logs"].keys())[0]]["validate_every"]], checkpoint_steps.astype(int) )
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title("Learning Rate during training steps")
        # plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/kappa.png")
        plt.show()