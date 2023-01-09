import torch
import torch.nn as nn
from colorama import init, Fore, Back, Style
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import numpy as np
from collections import defaultdict
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from scipy.special import softmax
import einops

class Validator():
    def __init__(self, model, data_loader, config, device):
        self.config = config
        self.device = device
        self.model = model
        self.data_loader = data_loader

    def get_results(self, set: str = "Validation", print_results: bool=False, show_border_info: bool=False):

        if set == "Validation":
            this_dataloader = self.data_loader.valid_loader
        elif set == "Test":
            this_dataloader = self.data_loader.test_loader
        elif set == "Train":
            this_dataloader = self.data_loader.train_loader
        elif set == "Total":
            this_dataloader = self.data_loader.total_loader
        else:
            raise ValueError('This set {} does not exist, options are "Validation", "Test", Train" "Total"'.format(set))

        metrics = self.validate(data_loader=this_dataloader, description=set, show_border_info=show_border_info)

        self.print_results(metrics=metrics, description=set)
        if print_results:
            self.norm_n_plot_confusion_matrix(metrics["conf"]["combined"], description=set)
        return metrics

    def _perf_measure(self, y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)):
            if y_actual[i] == y_hat[i] == 1:
                TP += 1
            if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
                FP += 1
            if y_actual[i] == y_hat[i] == 0:
                TN += 1
            if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
                FN += 1

        return (TP, FP, TN, FN)

    def validate(self, data_loader, description, show_border_info=False, skip_modality="full"):
            self.model.eval()
            with torch.no_grad():
                tts, preds, inits = [], [], []
                pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
                # for batch_idx, (data, target, init, _) in pbar:
                for batch_idx, served_dict in pbar:

                    served_dict["data"] = {view: served_dict["data"][view].float().to(self.device) for view in served_dict["data"]}
                    target = served_dict["label"][list(served_dict["label"].keys())[0]].flatten(start_dim=0, end_dim=1).to(self.device)

                    if "three_modes" in self.config.model.args and self.config.model.args.three_modes:
                        # pred = self.model(served_dict["data"],skip_modality="full", return_matches=True)
                        # pred = self.model(served_dict["data"], skip_modality=served_dict["skip_view"])
                        # pred["preds"]["skipped"] = self.model(served_dict["data"], skip_modality=served_dict["skip_view"])["preds"]["combined"]
                        # pred = self.get_predictions_time_series(served_dict["data"], served_dict["init"], skip_modality=served_dict["skip_view"])
                        if self.model.module._get_name() ==  'EEG_SLEEP_BLIP_GM_MultiMode':
                            pred = self.model(served_dict["data"], skip_modality=served_dict["skip_view"], return_matches=True)
                            # pred = self.model(served_dict["data"], skip_modality="full", return_matches=True)
                        else:
                            pred = self.model(served_dict["data"],skip_modality="full", return_matches=True)
                            pred["preds"]["skipped"] = self.get_predictions_time_series(served_dict["data"], served_dict["init"], skip_modality=served_dict["skip_view"])["preds"]["combined"]
                        # pred["preds"]["eeg"] = self.model(served_dict["data"], skip_modality="eog")["preds"]["combined"]
                        # pred["preds"]["eog"] = self.model(served_dict["data"], skip_modality="eeg")["preds"]["combined"]
                        if "stft_emg" in served_dict["data"]:
                            pred["preds"]["emg"] = self.model(served_dict["data"], skip_modality="emg")["preds"]["combined"]
                    else:
                        pred = self.model(served_dict["data"])
                    tts.append(target)
                    preds.append(pred["preds"])
                    pbar.set_description("{} batch {}/{}".format(description, int(batch_idx), int(len(data_loader))))
                    pbar.refresh()

                if "softlabels" in self.config.dataset and self.config.dataset.softlabels:
                    tts = torch.cat(tts).argmax(dim=1).cpu().numpy()
                else:
                    tts = torch.cat(tts).cpu().numpy()

                # preds = torch.cat(preds).cpu().numpy()

            multiclass = True

            total_preds, metrics = {}, defaultdict(dict)
            for pred_key in preds[0].keys():
                total_preds[pred_key] = np.concatenate([pred[pred_key].cpu().numpy() for pred in preds], axis=0)
                metrics["entropy"][pred_key] = entropy(softmax(total_preds[pred_key], axis=1), axis=1)
                total_preds[pred_key] = total_preds[pred_key].argmax(axis=-1)
                metrics["entropy_correct"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] == tts].mean()
                metrics["entropy_correct_var"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] == tts].std()
                metrics["entropy_wrong"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] != tts].mean()
                metrics["entropy_wrong_var"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] != tts].std()
                metrics["entropy_var"][pred_key] = metrics["entropy"][pred_key].std()
                metrics["entropy"][pred_key] = metrics["entropy"][pred_key].mean()
                metrics["acc"][pred_key] = np.equal(tts, total_preds[pred_key]).sum() / len(tts)
                metrics["f1"][pred_key] = f1_score(total_preds[pred_key], tts, average="macro")
                metrics["k"][pred_key] = cohen_kappa_score(total_preds[pred_key], tts)
                metrics["f1_perclass"][pred_key] = f1_score(total_preds[pred_key], tts, average=None)
                metrics["auc"][pred_key] = roc_auc_score(tts, total_preds[pred_key]) if not multiclass else 0
                metrics["conf"][pred_key] = confusion_matrix(tts, total_preds[pred_key])
                tp, fp, tn, fn = self._perf_measure(tts, preds)
                metrics["spec"][pred_key] = tn / (tn + fp) if (tn + fp) != 0 else 0
                metrics["sens"][pred_key] = tp / (tp + fn) if (tp + fn) != 0 else 0
                if show_border_info:
                    self.validate_borders(targets = tts, preds = total_preds[pred_key], description=description+" "+pred_key)
            metrics = dict(metrics)  # Avoid passing empty dicts to logs, better return an error!

            return metrics

    def get_predictions_time_series(self, views, inits, skip_modality):
        """
        This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
        :param views: List of tensors, data views/modalities
        :param inits: Tensor indicating with value one, when there incontinuities.
        :return: predictions of the self.model on the batch
        """
        this_inits = inits[list(inits.keys())[0]]
        this_view = views[list(inits.keys())[0]]
        skip_modality_temp = torch.zeros(this_view.shape[0], this_view.shape[1])
        skip_modality_temp[skip_modality["stft_eeg"] == 1] = 1
        skip_modality_temp[skip_modality["stft_eog"] == 1] = 2

        a = ((skip_modality_temp - skip_modality_temp.mean(dim=1).unsqueeze(dim=1).repeat(1, 21)).mean(dim=1) != 0).nonzero(as_tuple=True)[0]
        for batch_idx in a:
            for i in range(len(skip_modality_temp[batch_idx]) - 1):
                if skip_modality_temp[batch_idx][i] != skip_modality_temp[batch_idx][i + 1]:
                    this_inits[batch_idx][i] = 1
                    this_inits[batch_idx][i + 1] = 1

        inits_sum_batch = (this_inits.sum(dim=1) > 1).nonzero(as_tuple=True)[0]
        if len(inits_sum_batch) > 0:
            batch, outer = this_view.shape[0], this_view.shape[1]
            batch_idx_checked = torch.ones(batch, dtype=torch.bool)
            pred ={"preds":{"combined": torch.zeros(batch * outer, self.config.num_classes).to(this_view.device)}}
            for batch_idx in inits_sum_batch:
                ones_idx = (this_inits[batch_idx] > 0).nonzero(as_tuple=True)[0]
                if (ones_idx[0] + 1 == ones_idx[1]  ): #and ones_idx[0]!=0 and ones_idx[1]!= len(inits[idx])
                    if ones_idx[0] == 0:
                        pred_split_0 = self.model({view: views[view][batch_idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views}, skip_modality={view: skip_modality[view][batch_idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views})
                    else:
                        pred_split_0 = self.model({view: views[view][batch_idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views}, skip_modality={view: skip_modality[view][batch_idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views})

                    if ones_idx[1] == len(this_inits[batch_idx]):
                        pred_split_1 = self.model({view: views[view][batch_idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views}, skip_modality={view: skip_modality[view][batch_idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views})
                    else:
                        pred_split_1 = self.model({view: views[view][batch_idx, ones_idx[1]:].unsqueeze(dim=0) for view in views}, skip_modality={view: skip_modality[view][batch_idx, ones_idx[1]:].unsqueeze(dim=0) for view in views})

                    pred["preds"]["combined"][batch_idx * outer:(batch_idx + 1) * outer] = torch.cat([pred_split_0["preds"]["combined"], pred_split_1["preds"]["combined"]], dim=0)
                else:
                    pred["preds"]["combined"][batch_idx * outer:(batch_idx + 1) * outer] = self.model({view: views[view][batch_idx].unsqueeze(dim=0) for view in views}, skip_modality={view: skip_modality[view][batch_idx].unsqueeze(dim=0) for view in views})["preds"]["combined"]

                batch_idx_checked[batch_idx] = False
            pred["preds"]["combined"][batch_idx_checked.repeat_interleave(outer)] = self.model({view: views[view][batch_idx_checked] for view in views}, skip_modality={view: skip_modality[view][batch_idx_checked] for view in views})["preds"]["combined"]

        else:
            pred = self.model(views, skip_modality=skip_modality)

        return pred

    def validate_borders(self, targets, preds, description):

        tts =targets
        seq_nums = np.expand_dims(np.arange(self.config.dataset.seq_length[0]), axis=0).repeat((len(tts) // self.config.dataset.seq_length[0])+1, axis=0).flatten()[0:len(tts)]
        class_pred = preds

        border_points = [4]
        for i in range(1, len(tts) - 1):
            if tts[i - 1] != tts[i] or tts[i + 1] != tts[i]:
                border_points.append(0)
            else:
                border_points.append(4)
        border_points.append(4)
        for i in range(len(border_points)):
            if border_points[i] == 0:
                if border_points[i + 1] != 0:
                    border_points[i + 1] = 1
                if border_points[i - 1] != 0:
                    border_points[i - 1] = 1
                if border_points[i + 2] != 0 and border_points[i + 2] != 1:
                    border_points[i + 2] = 2
                if border_points[i - 2] != 0 and border_points[i - 2] != 1:
                    border_points[i - 2] = 2
                if border_points[i + 3] != 0 and border_points[i + 3] != 1 and border_points[i + 3] != 2:
                    border_points[i + 3] = 3
                if border_points[i - 3] != 0 and border_points[i - 3] != 1 and border_points[i - 3] != 2:
                    border_points[i - 3] = 3
        border_points = np.array(border_points)
        _, border_counts_p = np.unique(border_points, return_counts=True)
        _, border_counts = np.unique(border_points[~(class_pred == tts)], return_counts=True)
        border_counts = border_counts / border_counts.sum()
        border_colors = [(0, 0, 0.8), (0, 0, 0.6), (0, 0, 0.4), (0, 0, 0.2), "gold"]
        plt.bar(np.array([0, 1, 2, 3, 4]), border_counts, color=border_colors)
        plt.xticks(np.array([0, 1, 2, 3, 4]), labels=['Border', '1', '2', '3', 'Rest'])
        plt.ylabel("Percentage of Mistakes")
        plt.title("Border %Mistakes with {}/{} on {} SHHS ".format((~(class_pred == tts)).sum(),len(tts), description))
        plt.show()

        _, border_counts = np.unique(border_points, return_counts=True)
        print("We have in total {}".format(border_counts.sum()))
        print(border_counts)
        border_counts = border_counts / border_counts.sum()
        border_colors = [(0, 0, 0.8), (0, 0, 0.6), (0, 0, 0.4), (0, 0, 0.2), "gold"]
        plt.bar(np.array([0, 1, 2, 3, 4]), border_counts, color=border_colors)
        plt.xticks(np.array([0, 1, 2, 3, 4]), labels=['Border', '1', '2', '3', 'Rest'])
        plt.ylabel("%")
        plt.title("Border distribution on {} SHHS".format(description))
        plt.show()

        seq_colors = [(0, 0, 0),
                         (0, 0, 0.1),
                         (0, 0, 0.2),
                         (0, 0, 0.3),
                         (0, 0, 0.4),
                         (0, 0, 0.5),
                         (0, 0, 0.6),
                         (0, 0, 0.7),
                         (0, 0, 0.8),
                         (0, 0, 0.9),
                         (0, 0, 1),
                         (0, 0, 0.9),
                         (0, 0, 0.8),
                         (0, 0, 0.7),
                         (0, 0, 0.6),
                         (0, 0, 0.5),
                         (0, 0, 0.4),
                         (0, 0, 0.3),
                         (0, 0, 0.2),
                         (0, 0, 0.1),
                         (0, 0, 0)]

        _, seq_counts = np.unique(seq_nums[~(class_pred == tts)], return_counts=True)
        seq_counts = seq_counts / seq_counts.sum()
        x = np.arange(self.config.dataset.seq_length[0])
        plt.bar(x, seq_counts, color=seq_colors)
        plt.ylabel("Percentage of Mistakes")
        plt.title("Seq Num mistakes on {} SHHS".format(description))
        plt.show()

    def validate_ensembles(models, data_loader, description):
        raise NotImplementedError()
        ensemble_preds = []
        for model in models:
            model.eval()
            with torch.no_grad():
                tts, preds, inits = [], [], []
                pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
                for batch_idx, (data, target, init, _) in pbar:
                    views = [data[i].float().to(device) for i in range(len(data))]
                    label = target.to(device).flatten()
                    pred = get_predictions_time_series(model, views, init)
                    tts.append(label)
                    preds.append(pred)
                    inits.append(init.flatten())
                    pbar.set_description("{} batch {}/{}".format(description, int(batch_idx), int(len(data_loader))))
                    pbar.refresh()

                tts = torch.cat(tts).cpu().numpy()
                preds = torch.cat(preds).cpu().numpy()
            ensemble_preds.append(preds)

        for i in range(len(models)):
            print("--------------------------------------------")
            print("Model {} has ".format(i))
            preds = np.array(ensemble_preds[i])

            multiclass = False
            if preds.shape[1] > 2:
                multiclass = True

            entropy_pred = entropy(preds, axis=1)
            class_pred = preds.argmax(axis=1)
            entropy_correct_class = entropy_pred[class_pred == tts].mean()
            entropy_wrong_class = entropy_pred[class_pred != tts].mean()

            print("{} entropy for correct class class prediction is {} and for wrong class predictions {}".format(
                description, entropy_correct_class, entropy_wrong_class))

            preds = preds.argmax(axis=1)
            test_acc = np.equal(tts, preds).sum() / len(tts)
            test_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average="macro")
            test_perclass_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average=None)
            test_k = cohen_kappa_score(tts, preds)
            test_auc = roc_auc_score(tts, preds) if not multiclass else 0
            test_conf = confusion_matrix(tts, preds)
            tp, fp, tn, fn = perf_measure(tts, preds)
            test_spec = tn / (tn + fp) if (tn + fp) != 0 else 0
            test_sens = tp / (tp + fn) if (tp + fn) != 0 else 0
            print(
                "{0} accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, sens:{4:.4f}, spec:{5:.4f}, f1_per_class :{6:40}".format(
                    description,
                    test_acc * 100,
                    test_f1,
                    test_k, test_spec, test_sens,
                    "{}".format(list(test_perclass_f1))))
            norm_n_plot_confusion_matrix(test_conf, description)

        print("--------------------------------------------")
        print("Ensemble of Models")

        preds = np.array(ensemble_preds).mean(axis=0)

        multiclass = False
        if preds.shape[1] > 2:
            multiclass = True

        entropy_pred = entropy(preds, axis=1)
        class_pred = preds.argmax(axis=1)
        entropy_correct_class = entropy_pred[class_pred == tts].mean()
        entropy_wrong_class = entropy_pred[class_pred != tts].mean()

        print(
            "{} entropy for correct class class prediction is {} and for wrong class predictions {}".format(description,
                                                                                                            entropy_correct_class,
                                                                                                            entropy_wrong_class))

        preds = preds.argmax(axis=1)
        test_acc = np.equal(tts, preds).sum() / len(tts)
        test_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average="macro")
        test_perclass_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average=None)
        test_k = cohen_kappa_score(tts, preds)
        test_auc = roc_auc_score(tts, preds) if not multiclass else 0
        test_conf = confusion_matrix(tts, preds)
        tp, fp, tn, fn = perf_measure(tts, preds)
        test_spec = tn / (tn + fp) if (tn + fp) != 0 else 0
        test_sens = tp / (tp + fn) if (tp + fn) != 0 else 0
        print("{0} accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, sens:{4:.4f}, spec:{5:.4f}, f1_per_class :{6:40}".format(
            description,
            test_acc * 100,
            test_f1,
            test_k, test_spec, test_sens,
            "{}".format(list(test_perclass_f1))))
        norm_n_plot_confusion_matrix(test_conf, description)

        return test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens

    def print_results(self, metrics, description):

        for i, v in metrics["acc"].items():
            message = Style.BRIGHT + Fore.WHITE + "{} ".format(description)
            latex_message = Style.BRIGHT + Fore.WHITE + "{} ".format(description)
            message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i, metrics["acc"][i] * 100)
            latex_message += " {:.1f} &".format(metrics["acc"][i] * 100)
            message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i, metrics["k"][i])
            latex_message += " {:.3f} &".format(metrics["k"][i])
            message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i, metrics["f1"][i] * 100)
            latex_message += " {:.1f} &".format(metrics["f1"][i] * 100)
            message += Fore.BLUE + "F1_perclass_{}: {} ".format(i,"{}".format(str(list((metrics["f1_perclass"][i] * 100).round(2)))))
            for i in list((metrics["f1_perclass"][i] * 100).round(2)):
                latex_message += " {:.1f} &".format(i)
            print(message+ Style.RESET_ALL)
            print(latex_message+ Style.RESET_ALL)

        # message = ""
        # for i, v in metrics["entropy"].items(): message += Fore.LIGHTRED_EX + "E_{}: {:.4f} ".format(i, v)
        # for i, v in metrics["entropy_var"].items(): message += Fore.LIGHTRED_EX + "E_var_{}: {:.4f} ".format(i, v)
        # for i, v in metrics["entropy_correct"].items(): message += Fore.LIGHTMAGENTA_EX + "EC_{}: {:.4f} ".format(i, v)
        # for i, v in metrics["entropy_correct_var"].items(): message += Fore.LIGHTMAGENTA_EX + "EC_var_{}: {:.4f} ".format(i, v)
        # for i, v in metrics["entropy_wrong"].items(): message += Fore.LIGHTYELLOW_EX + "EW_{}: {:.4f} ".format(i, v)
        # for i, v in metrics["entropy_wrong_var"].items(): message += Fore.LIGHTYELLOW_EX + "EW_var_{}: {:.4f} ".format(i, v)
        # print(message+ Style.RESET_ALL)

    def save_test_results(self, checkpoint, save_dir, test_results, skipped=False):

        test_results_dict = { "post_test_results_skipped": test_results} if skipped else { "post_test_results": test_results}
        checkpoint.update(test_results_dict)
        try:
            torch.save(checkpoint, save_dir)
            print("Models has saved successfully in {}".format(save_dir))
        except:
            raise Exception("Problem in model saving")

    def norm_n_plot_confusion_matrix(self, test_conf, description):
        test_conf = test_conf.astype(float)
        for j in range(len(test_conf)):
            th_sum = test_conf[j].sum()
            for i in range(len(test_conf[j])):
                test_conf[j][i] /= th_sum
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        print(test_conf)
        test_conf = np.round(test_conf, 3)
        import seaborn as sns
        import matplotlib.pyplot as plt

        ax = plt.subplot()
        sns.heatmap(test_conf, annot=True, fmt='g', ax=ax)

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('{} Confusion Matrix'.format(description))
        ax.xaxis.set_ticklabels(['Wake', 'N1','N2', 'N3', 'REM'])
        ax.yaxis.set_ticklabels(['REM', 'N3','N2', 'N1', 'Wake'])
        plt.show()

