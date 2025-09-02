import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


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
        plt.show()

    def sleep_plot_losses(self, chosen_loss = None):

        list_losses = ["total"]
        if "multi_loss" in self.config.model.args:
            list_losses += ["ce_loss_{}".format(i) for i, v in self.config.model.args.multi_loss.multi_supervised_w.items() if v != 0]
            list_losses += [i for i, v in self.config.model.args.multi_loss.items() if v != 0 and type(v) == int]

        if type(self.logs["train_logs"][list(self.logs["train_logs"].keys())[0]]["train_loss"]) != dict: return self.sleep_plot_losses_old()

        train_loss = { loss_key: np.array([self.logs["train_logs"][i]["train_loss"][loss_key] for i in self.logs["train_logs"]]) for loss_key in list_losses}
        val_loss = {loss_key: np.array([self.logs["val_logs"][i]["val_loss"][loss_key] for i in self.logs["val_logs"]]) for loss_key in list_losses}
        if "test_logs" in self.logs:
            test_loss = {loss_key: np.array([self.logs["test_logs"][i]["test_loss"][loss_key] for i in self.logs["test_logs"]]) for loss_key in list_losses}


        if chosen_loss and chosen_loss not in list_losses: raise ValueError("chosen loss doesnt exist, choose from {}".format(list_losses[1:]))

        steps = np.array([i / self.logs["train_logs"][i]["validate_every"] for i in self.logs["train_logs"]]) - 1
        best_step = (self.logs["best_logs"]["step"] / self.logs["train_logs"][self.logs["best_logs"]["step"]]["validate_every"]) - 1

        plt.figure()
        loss_min = 100
        loss_max = 0
        plotted_losses = 0
        for loss_key in list_losses:
            if loss_key == "total" or (chosen_loss and chosen_loss!=loss_key): continue

            plt.plot(steps, (train_loss[loss_key]), label="Train_{}".format(loss_key))
            plt.plot(steps, (val_loss[loss_key]), label="Valid_{}".format(loss_key))
            if "test_logs" in self.logs and len(test_loss[loss_key])>0:
                plt.plot(np.arange(0, len(steps), len(steps)/len(test_loss[loss_key])), test_loss[loss_key], label="Test_{}".format(loss_key))

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
            plt.show()
        else:
            plt.close()

        plt.figure()
        loss_min = 100
        loss_max = 0
        loss_key = "total"
        plt.plot(steps, train_loss[loss_key], label="Train_{}".format(loss_key))
        plt.plot(steps, val_loss[loss_key], label="Valid_{}".format(loss_key))
        if "test_logs" in self.logs and len(test_loss[loss_key])>0:
            plt.plot(np.arange(0, len(steps), len(steps) / len(test_loss[loss_key])), test_loss[loss_key],
                     label="Test_{}".format(loss_key))

            # plt.plot(steps, test_loss[loss_key], label="Test_{}".format(loss_key))
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
        plt.show()

    def plot_unimulti_performance(self):

        if "multi_loss" in self.config.model.args:
            list_predictors = ["{}".format(i) for i, v in self.config.model.args.multi_loss.multi_supervised_w.items() if v != 0]
        else:
            list_predictors = ["combined"]

        if "cca" in list_predictors: list_predictors.remove("cca")
        val_acc = {pred_key: np.array([self.logs["val_logs"][i]["val_acc"][pred_key] for i in self.logs["val_logs"]]) for pred_key in list_predictors}
        train_acc = {pred_key: np.array([self.logs["train_logs"][i]["train_acc"][pred_key] for i in self.logs["train_logs"]]) for pred_key in list_predictors}

        # test_acc = {pred_key: np.array([self.logs["test_logs"][i]["test_acc"][pred_key] for i in self.logs["test_logs"]]) for pred_key in list_predictors}
        # test_steps = (np.array(list(self.logs["test_logs"].keys()))/self.logs["train_logs"][list(self.logs["test_logs"].keys())[0]]["validate_every"]).astype(int) -1

        steps = np.array([i / self.logs["train_logs"][i]["validate_every"] for i in self.logs["train_logs"]]) - 1
        best_step = (self.logs["best_logs"]["step"] / self.logs["train_logs"][self.logs["best_logs"]["step"]]["validate_every"])-1

        #
        # plt.figure()
        loss_min = 100
        loss_max = 0
        # print(list_predictors)
        # a = ["combined"] if "c" not in list_predictors else ["c"]
        # print(a)
        for loss_key in list_predictors:

            plt.plot(steps, train_acc[loss_key], label="Train {}".format(loss_key))
            plt.plot(steps, val_acc[loss_key], label="Val {}".format(loss_key))
            # plt.plot(test_steps, test_acc[loss_key], label="Test {}".format(loss_key))
            #
            loss_min = np.minimum(loss_min, np.min(val_acc[loss_key]))
            loss_max = np.maximum(loss_max, np.max(val_acc[loss_key]))
            best_loss = {loss_key: self.logs["best_logs"]["val_f1"][loss_key]}
            plt.plot((best_step, best_step), (0, best_loss[loss_key]), linestyle="--", color="y", label="Chosen Point")
            plt.plot((0, best_step), (best_loss[loss_key], best_loss[loss_key]), linestyle="--", color="y")
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        # plt.title("F1 {} Predictors".format(loss_key))
        plt.title("Validation: Multimodal and Unimodal predictors")
        # plt.ylim([loss_min - 0.1, loss_max + 0.1])
        plt.legend()
        plt.show()

    def plot_regularizers(self):

        if "reg_logs" not in self.logs: return
        self.reg_logs = self.logs["reg_logs"]

        window_size = 300
        order = 1
        if len(self.reg_logs["coeff"])>window_size:
            plt.plot(savgol_filter(np.array(self.reg_logs["coeff"]), window_length=window_size, polyorder=order))
            plt.title("Magnitude correction coefficient vision/audio")
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            # plt.legend(fontsize=10, loc="upper left")
            plt.show()

        plt.plot(savgol_filter(self.reg_logs["norm_m_g"], window_length=window_size, polyorder=order), color="blue", label="Multimodal Vision")
        plt.plot(savgol_filter(self.reg_logs["norm_m_c"], window_length=window_size, polyorder=order), color="green", label="Multimodal Audio")
        plt.plot(savgol_filter(self.reg_logs["postcorr_norm_m_g"], window_length=window_size, polyorder=order), color="yellow", label="Multimodal Corr Vision")
        plt.plot(savgol_filter(self.reg_logs["postcorr_norm_m_c"], window_length=window_size, polyorder=order), color="red", label="Multimodal Corr Audio")
        plt.plot(savgol_filter(self.reg_logs["norm_g"], window_length=window_size, polyorder=order), color="blue",  linestyle='--',label="Unimodal Vision")
        plt.plot(savgol_filter(self.reg_logs["norm_c"], window_length=window_size, polyorder=order), color="green", linestyle='--', label="Unimodal Audio")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.legend(fontsize=10, loc="upper left")
        plt.xlabel("Optimization Steps")
        plt.ylabel("Average Gradient Norm")
        plt.title("Gradient Norm Multimodal-Unimodal")
        plt.show()

        plt.plot(savgol_filter(self.reg_logs["postcorr_g"], window_length=window_size, polyorder=order), color="blue", label="Vision Branch")
        plt.plot(savgol_filter(self.reg_logs["postcorr_c"], window_length=window_size, polyorder=order), color="green", label="Audio Branch")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.legend(fontsize=10, loc="upper left")
        plt.xlabel("Optimization Steps")
        plt.ylabel("Average Gradient Norm")
        plt.title("Gradient Norm of each Modality")
        plt.show()

        plt.plot(savgol_filter(self.reg_logs["angle_m_g"], window_length=window_size, polyorder=order), color="blue", label="Vision Branch")
        plt.plot(savgol_filter(self.reg_logs["angle_m_c"], window_length=window_size, polyorder=order), color="green", label="Audio Branch")
        plt.plot(savgol_filter(self.reg_logs["postcorr_angle_m_g"], window_length=window_size, polyorder=order), color="lightblue", label="Post Audio Branch")
        plt.plot(savgol_filter(self.reg_logs["postcorr_angle_m_c"], window_length=window_size, polyorder=order), color="lightgreen", label="Post Video Branch")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.legend(fontsize=10, loc="upper left")
        plt.xlabel("Optimization Steps")
        plt.ylabel("Average Gradient Angles")
        plt.title("Gradient Angle of each Unimodal-Multimodal loss")
        plt.show()

        return

        # total_reg = savgol_filter(np.array(self.reg_logs["total_reg"]).flatten(), window_length=window_size, polyorder=order)

        enc_0_a = savgol_filter(np.array(self.reg_logs["enc_0_a"]).flatten(), window_length=window_size, polyorder=order)
        enc_1_a = savgol_filter(np.array(self.reg_logs["enc_1_a"]).flatten(), window_length=window_size, polyorder=order)
        enc_0_v = savgol_filter(np.array(self.reg_logs["enc_0_v"]).flatten(), window_length=window_size, polyorder=order)
        enc_1_v = savgol_filter(np.array(self.reg_logs["enc_1_v"]).flatten(), window_length=window_size, polyorder=order)

        enc_0_t = savgol_filter(np.array(self.reg_logs["enc_0_t"]).flatten(), window_length=window_size, polyorder=order)
        enc_1_t = savgol_filter(np.array(self.reg_logs["enc_1_t"]).flatten(), window_length=window_size, polyorder=order)


        x = np.arange(0, len(enc_0_a))
        plt.figure(figsize=(10, 6))  # Set a larger figure size

        plt.plot(x, enc_0_a, color="blue", label="Audio -> Enc_A")
        plt.plot(x, enc_1_v, color="lightblue", label="Vision -> Enc_V")
        plt.plot(x, enc_1_a, color="blue", linestyle="--", label="Audio -> Enc_V")
        plt.plot(x, enc_0_v, color="lightblue", linestyle="--", label="Vision -> Enc_A")
        plt.title("Gradient Norm of Shapley Values", fontsize=14)
        plt.ylabel("Norm of Gradient", fontsize=12)
        plt.xlabel("Optimization Steps", fontsize=12)
        plt.xticks(np.arange(min(x), max(x) + 1, 5000))
        plt.axvline(self.logs["best_logs"]["step"], label="Best Model", color="red")

        plt.yticks(np.arange(0,max(enc_0_a), 10))#[0,10,20,30,40,50,60,70,80])
        plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # plt.axis("off")
        plt.legend(fontsize=10, loc="upper left")
        plt.show()

        x = np.arange(0, len(enc_0_t))
        plt.figure(figsize=(10, 6))  # Set a larger figure size

        plt.plot(x, enc_0_t, color="blue", label="Enc_A")
        plt.plot(x, enc_1_t, color="lightblue", label="Enc_V")
        plt.title("Gradient Norm", fontsize=14)
        plt.ylabel("Norm of Gradient", fontsize=12)
        plt.xlabel("Optimization Steps", fontsize=12)
        plt.xticks(np.arange(min(x), max(x) + 1, 5000))
        plt.axvline(self.logs["best_logs"]["step"], label="Best Model", color="red")

        plt.yticks(np.arange(0,max(enc_0_a), 10))#[0,10,20,30,40,50,60,70,80])
        plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # plt.axis("off")
        plt.legend(fontsize=10, loc="upper left")
        plt.show()


        # reg_cg = savgol_filter(np.array(self.reg_logs["reg_cg"]).flatten(), window_length=window_size, polyorder=order)
        # reg_cm = savgol_filter(np.array(self.reg_logs["reg_cm"]).flatten(), window_length=window_size, polyorder=order)
        # reg_gm = savgol_filter(np.array(self.reg_logs["reg_gm"]).flatten(), window_length=window_size, polyorder=order)
        #
        # reg_cos_cg = savgol_filter(np.array(self.reg_logs["reg_cos_cg"]).flatten(), window_length=window_size, polyorder=order)
        # reg_cos_cm = savgol_filter(np.array(self.reg_logs["reg_cos_cm"]).flatten(), window_length=window_size, polyorder=order)
        # reg_cos_gm = savgol_filter(np.array(self.reg_logs["reg_cos_gm"]).flatten(), window_length=window_size, polyorder=order)
        #
        # grad_norm_c = savgol_filter(np.array(self.reg_logs["grad_norm_c"]).flatten(), window_length=window_size, polyorder=order)
        # grad_norm_g = savgol_filter(np.array(self.reg_logs["grad_norm_g"]).flatten(), window_length=window_size, polyorder=order)
        # grad_norm_m = savgol_filter(np.array(self.reg_logs["grad_norm_m"]).flatten(), window_length=window_size, polyorder=order)
        #
        #
        # up_to = 6000
        #
        # plt.figure()
        # plt.plot(total_reg[:up_to], alpha=0.7)
        # plt.xlabel("Opt Steps")
        # plt.title("Total Mean Regularization {:.2f}".format(total_reg.mean()))
        # plt.show()

        # plt.figure()
        # plt.plot(reg_cg[:up_to], label="Dist CG", alpha=0.7)
        # plt.plot(reg_cm[:up_to], label="Dist CM", alpha=0.7)
        # plt.plot(reg_gm[:up_to], label="Dist GM", alpha=0.7)
        # plt.legend()
        # plt.xlabel("Opt Steps")
        # plt.ylabel("Norm")
        # plt.title("Norm of Gradient Distance CG {:.2f}, CM {:.2f} and GM {:.2f}".format(reg_cg[:up_to].mean(),reg_cm[:up_to].mean(), reg_gm[:up_to].mean()))
        # plt.show()
        # #
        # plt.figure()
        # plt.plot(reg_cos_cg, label="Angle CG", alpha=0.7)
        # plt.plot(reg_cos_cm, label="Angle CM", alpha=0.7)
        # plt.plot(reg_cos_gm, label="Angle GM", alpha=0.7)
        # plt.legend()
        # plt.xlabel("Opt Steps")
        # plt.ylabel("Cossine Similarity")
        # plt.title("Cossine Similarity of Gradients CG {:.2f}, CM {:.2f} and GM {:.2f}".format(reg_cos_cg.mean(),reg_cos_cm.mean(), reg_cos_gm.mean()))
        # plt.show()
        #
        # plt.figure()
        # plt.plot(grad_norm_c, label="Grad C", alpha=0.7)
        # plt.plot(grad_norm_g, label="Grad G", alpha=0.7)
        # plt.plot(grad_norm_m, label="Grad M", alpha=0.7)
        # # plt.plot(total_reg/total_reg.mean(), label="Reg", alpha=0.7)
        # plt.legend()
        # plt.xlabel("Opt Steps")
        # plt.ylabel("Norm of Gradient")
        # plt.title("Norm of Gradient C {:.2f}, G {:.2f} and M {:.2f}".format(reg_cg.mean(),reg_cm.mean(), reg_gm.mean()))
        # plt.show()

    def plot_ratios(self):



        if "ratio_logs" not in self.logs: return
        this_ratio = {ratio: np.array(self.logs["ratio_logs"][ratio]).flatten() for ratio in self.logs["ratio_logs"]}

        window_size = 101
        order = 1

        this_ratio = {n: savgol_filter(this_ratio[n], window_length=window_size, polyorder=order) for n in this_ratio}

        plt.figure()
        plt.plot(np.array(this_ratio["ratio_PMR"]), label="Ratio Color")
        # plt.plot(np.array(this_ratio["ratio_color"]), label="Ratio Color")
        # plt.plot(np.array(this_ratio["ratio_gray"]), label="Ratio Gray")
        plt.legend()
        plt.xlabel("Opt Steps")
        plt.ylabel("Ratio Value")
        plt.title("Ratio Color/Gray {:.2f}".format(np.array(this_ratio["ratio_PMR"]).mean()))
        # plt.title("Ratio Color/Gray {:.2f}".format(np.array(this_ratio["ratio_color"]).mean()))
        plt.show()

        # plt.figure()
        # plt.plot(np.array(this_ratio["coeff_color"]), label="Color")
        # plt.plot(np.array(this_ratio["coeff_gray"]), label="Gray")
        # plt.legend()
        # plt.xlabel("Opt Steps")
        # plt.ylabel("Coeff Value")
        # plt.title("Coeff Color {:.2f} - Gray {:.2f}".format(np.array(this_ratio["coeff_color"]).mean(), np.array(this_ratio["coeff_gray"]).mean()))
        # plt.show()

    def sleep_plot_performance(self):

        if "multi_loss" in self.config.model.args:
            list_predictors = ["{}".format(i) for i, v in self.config.model.args.multi_loss.multi_supervised_w.items() if v != 0]
        else:
            list_predictors = ["combined"]

        if type(self.logs["train_logs"][list(self.logs["train_logs"].keys())[0]]["f1"]) != dict: return self.sleep_plot_performance_old()

        train_f1 = {pred_key: np.array([self.logs["train_logs"][i]["f1"][pred_key] for i in self.logs["train_logs"]]) for pred_key in list_predictors}
        val_f1 = {pred_key: np.array([self.logs["val_logs"][i]["f1"][pred_key] for i in self.logs["val_logs"]]) for pred_key in list_predictors}

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
            best_loss = {loss_key: self.logs["best_logs"]["f1"][loss_key]}
            plt.plot((best_step, best_step), (0, best_loss[loss_key]), linestyle="--", color="y", label="Chosen Point")
            plt.plot((0, best_step), (best_loss[loss_key], best_loss[loss_key]), linestyle="--", color="y")
            plt.xlabel('Steps')
            plt.ylabel('F1 Value')
            plt.title("F1 {} Predictors".format(loss_key))
            plt.ylim([loss_min - 0.05, loss_max + 0.05])
            plt.legend()
            plt.show()

        train_k = {pred_key: np.array([self.logs["train_logs"][i]["k"][pred_key] for i in self.logs["train_logs"]]) for pred_key in list_predictors}
        val_k = {pred_key: np.array([self.logs["val_logs"][i]["k"][pred_key] for i in self.logs["val_logs"]]) for pred_key in list_predictors}

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
            best_loss = {loss_key: self.logs["best_logs"]["k"][loss_key]}
            plt.plot((best_step, best_step), (0, best_loss[loss_key]), linestyle="--", color="y", label="Chosen Point")
            plt.plot((0, best_step), (best_loss[loss_key], best_loss[loss_key]), linestyle="--", color="y")
            plt.xlabel('Steps')
            plt.ylabel('K Value')
            plt.title("K {} Predictors".format(loss_key))
            plt.ylim([loss_min - 0.05, loss_max + 0.05])
            plt.legend()
            plt.show()

        train_acc = {pred_key: np.array([self.logs["train_logs"][i]["acc"][pred_key] for i in self.logs["train_logs"]]) for pred_key in list_predictors}
        val_acc = {pred_key: np.array([self.logs["val_logs"][i]["acc"][pred_key] for i in self.logs["val_logs"]]) for pred_key in list_predictors}

        for loss_key in list_predictors:
            plt.figure()
            loss_min = 100
            loss_max = 0
            plt.plot(steps, train_acc[loss_key], label="Train_{}".format(loss_key))
            plt.plot(steps, val_acc[loss_key], label="Valid_{}".format(loss_key))
            loss_min = np.minimum(loss_min, np.min(train_acc[loss_key]))
            loss_min = np.minimum(loss_min, np.min(val_acc[loss_key]))
            loss_max = np.maximum(loss_max, np.max(train_acc[loss_key]))
            loss_max = np.maximum(loss_max, np.max(val_acc[loss_key]))
            best_loss = {loss_key: self.logs["best_logs"]["acc"][loss_key]}
            plt.plot((best_step, best_step), (0, best_loss[loss_key]), linestyle="--", color="y", label="Chosen Point")
            plt.plot((0, best_step), (best_loss[loss_key], best_loss[loss_key]), linestyle="--", color="y")
            plt.xlabel('Steps')
            plt.ylabel('Accuracy')
            plt.title("Accuracy {} Predictors".format(loss_key))
            plt.ylim([loss_min - 0.05, loss_max + 0.05])
            plt.legend()
            plt.show()

        # train_f1_perclass = { pred_key: np.array([self.logs["train_logs"][i]["perclassf1"][pred_key] for i in self.logs["train_logs"]]) for pred_key in list_predictors}
        # val_f1_perclass = { pred_key: np.array([self.logs["val_logs"][i]["perclassf1"][pred_key] for i in self.logs["val_logs"]]) for pred_key in list_predictors}

        # for pred_key in list_predictors:
        #
        #     plt.figure()
        #     score_min = 100
        #     score_max = 0
        #     colors = ["b", "k", "r"]
        #     color_dict = {v: colors[i] for i, v in enumerate(list_predictors)}
        #     color_dict = {"Training":"b", "Validation":"r"}
        #     for set in [{"score": train_f1_perclass, "label": "Training"},
        #                 {"score": val_f1_perclass, "label": "Validation"}]:
        #
        #         plt.plot(steps, set["score"][pred_key][:, 0], color=color_dict[set["label"]], label="{}".format(set["label"]), linewidth=0.4)
        #         plt.plot(steps, set["score"][pred_key][:, 1], color=color_dict[set["label"]], linewidth=0.4)
        #         plt.plot(steps, set["score"][pred_key][:, 2], color=color_dict[set["label"]], linewidth=0.4)
        #         plt.plot(steps, set["score"][pred_key][:, 3], color=color_dict[set["label"]], linewidth=0.4)
        #         plt.plot(steps, set["score"][pred_key][:, 4], color=color_dict[set["label"]], linewidth=0.4)
        #
        #         score_min = np.minimum(score_min, np.min(set["score"][pred_key]))
        #         score_max = np.maximum(score_max, np.max(set["score"][pred_key]))
        #         if set["label"] == "Validation":
        #             for i in range(5):
        #                 best_loss = self.logs["best_logs"]["val_perclassf1"][pred_key][i]
        #                 plt.plot((best_step, best_step), (0, best_loss), linestyle="--", color="y", linewidth=0.6)
        #                 plt.plot((0, best_step), (best_loss, best_loss), linestyle="--", color="y", linewidth=0.6)
        #         else:
        #             plt.plot((best_step, best_step), (0, score_max), linestyle="--", color="y", linewidth=0.6)
        #
        #     plt.plot((0, steps[-1]), (0.8, 0.8), linestyle="--", linewidth=0.4, color="k")
        #     plt.plot((0, steps[-1]), (0.85, 0.85), linestyle="--", linewidth=0.4, color="k")
        #     plt.plot((0, steps[-1]), (0.9, 0.9), linestyle="--", linewidth=0.4, color="k")
        #     plt.plot((0, steps[-1]), (0.95, 0.95), linestyle="--", linewidth=0.4, color="k")
        #
        #     plt.xlabel('Steps')
        #     plt.ylabel('F1 Value')
        #     plt.title("F1 Multi Predictors on {}".format(set["label"]))
        #     plt.yticks([0.4, 0.45, 0.5, 0.55, 0.8, 0.85, 0.9, 0.95])
        #     plt.ylim([score_min - 0.05, score_max + 0.05])
        #     plt.legend()
        #     plt.show()

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
        plt.show()