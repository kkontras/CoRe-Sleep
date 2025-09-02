import torch
import torch.nn as nn
from colorama import init, Fore, Back, Style
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, top_k_accuracy_score
import numpy as np
from collections import defaultdict
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from scipy.special import softmax
import einops
# from torchdistill.core.forward_hook import ForwardHookManager
import seaborn as sns
# from utils.loss.CCA_Loss import CCA_Loss
import copy
import torchmetrics
import pickle
from torchmetrics import F1Score, CohenKappa, Accuracy
def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

class Validator():
    def __init__(self, config, device, model=None, data_loader=None):
        self.config = config
        self.device = device
        self.model = model
        self.data_loader = data_loader

    def get_set(self, set):
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
        return this_dataloader

    def get_results(self, set: str = "Validation", print_results: bool=False, show_border_info: bool=False):

        this_dataloader = self.get_set(set)
        metrics = self.validate(data_loader=this_dataloader, description=set, show_border_info=show_border_info)
        self.print_results(metrics=metrics, description=set)
        # if print_results:
        #     self.norm_n_plot_confusion_matrix(metrics["conf"]["combined"], description=set)
        return metrics

    def get_features(self, set: str = "Validation"):
        this_dataloader = self.get_set(set)
        self.model.eval()
        self.model.cuda()
        with torch.no_grad():
            tts, features, inits = [], defaultdict(list), []
            pbar = tqdm(enumerate(this_dataloader), total=len(this_dataloader), desc="Calculating Features", leave=False)
            # for batch_idx, (data, target, init, _) in pbar:
            for batch_idx, served_dict in pbar:

                # data = {view: served_dict["data"][view].cuda().float() for view in
                #         served_dict["data"] if type(served_dict["data"][view]) is torch.Tensor}
                #
                # data.update({view: data[view].float().cuda() for view in data if type(view) == int})

                data = {view: served_dict["data"][view].float().cuda() for view in served_dict["data"]}
                target = served_dict["label"].flatten(start_dim=0, end_dim=1).cuda()

                # target = served_dict["label"].type(torch.LongTensor).cuda()

                output = self.model(data, return_features=True)

                for i in output["features"]:
                    features[i].append(output["features"][i].cpu().detach())

                tts.append(target.cpu())


        for key in features:
            features[key] = torch.concat(features[key])
        tts = torch.concat(tts)

        #save in a file that will open up the fastest both features and tts
        torch.save({"features":features, "labels":tts}, "pretrained_feature_dataset_{}.pt".format(set))




        return features, tts

    def get_results_RF(self, model, set: str = "Validation", print_results: bool=False, show_border_info: bool=False):

        this_dataloader = self.get_set(set)
        metrics = self.validate_RF(model=model, data_loader=this_dataloader, description=set, show_border_info=show_border_info)
        self.print_results(metrics=metrics, description=set)
        return metrics




    def get_results_adv(self, set: str = "Validation", print_results: bool=False, show_border_info: bool=False):

        this_dataloader = self.get_set(set)
        metrics = self.validate_adv(data_loader=this_dataloader, description=set, show_border_info=show_border_info)
        self.print_results(metrics=metrics, description=set)
        if print_results:
            self.norm_n_plot_confusion_matrix(metrics["conf"]["combined"], description=set)
        return metrics

    def get_attention(self, set: str = "Validation", print_results: bool=False, show_border_info: bool=False):

        this_dataloader = self.get_set(set)
        metrics = self.attend(data_loader=this_dataloader, description=set, show_border_info=show_border_info)

        return metrics

    def get_cca(self, set: str = "Validation", print_results: bool=False, show_border_info: bool=False):

        this_dataloader = self.get_set(set)
        metrics = self.cca(data_loader=this_dataloader, description=set, show_border_info=show_border_info)

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
    def _get_loss_weights(self):

        if ("multi_loss" in self.config.model.args and "renew_each_step" in self.config.model.args.multi_loss and self.config.model.args.multi_loss.renew_each_step):
            w_loss = defaultdict(int)
            if "multi_loss" in self.config.model.args and "multi_loss_weights" in self.config.model.args.multi_loss:

                if "multi_supervised_loss" in self.config.model.args.multi_loss.multi_loss_weights:
                    for k, v in self.config.model.args.multi_loss.multi_loss_weights.multi_supervised_loss.items():
                        w_loss[k] = v
                w_loss["alignments"] = self.config.model.args.multi_loss.multi_loss_weights["alignment_loss"] if "alignment_loss" in self.config.model.args.multi_loss.multi_loss_weights else 0
                w_loss["order"] = self.config.model.args.multi_loss.multi_loss_weights["order_loss"] if "order_loss" in self.config.model.args.multi_loss.multi_loss_weights else 0
                w_loss["consistency"] = self.config.model.args.multi_loss.multi_loss_weights["consistency_loss"] if "consistency_loss" in self.config.model.args.multi_loss.multi_loss_weights else 0
                w_loss["reconstruction"] = self.config.model.args.multi_loss.multi_loss_weights["reconstruction"] if "reconstruction" in self.config.model.args.multi_loss.multi_loss_weights else 0
            else:
                w_loss["total"]= 1
                # raise Warning("We dont have multi supervised loss weights")
            self.w_loss = w_loss
        print("Loss Weights are", dict(self.w_loss))

    def ceu(self, total_preds, targets_tens):
        def create_conf(predictions):

            predictions = np.array(predictions)
            all_false = np.all(predictions[:2] == 0, axis=0)
            only_mod0_true = (predictions[0] == 1) & (predictions[1] == 0)
            only_mod1_true = (predictions[1] == 1) & (predictions[0] == 0)
            both_mods_true = (predictions[1] == 1) & (predictions[0] == 1)
            mmodel_true = predictions[2] == 1

            cm = np.array([
                [(~mmodel_true[all_false]).sum(), mmodel_true[all_false].sum()],
                [(~mmodel_true[only_mod0_true]).sum(), mmodel_true[only_mod0_true].sum()],
                [(~mmodel_true[only_mod1_true]).sum(), mmodel_true[only_mod1_true].sum()],
                [(~mmodel_true[both_mods_true]).sum(), mmodel_true[both_mods_true].sum()],
            ])
            mmodel_true[both_mods_true].sum()
            cm = 100 * cm.astype('float') / cm.sum()  # Normalize by row
            return cm

        this_fold = self.config.dataset.fold

        audio_preds = self.multi_fold_results[this_fold]["total_preds"]["combined"]
        audio_targets = self.multi_fold_results[this_fold]["total_preds_target"]
        video_preds = self.multi_fold_results[this_fold+3]["total_preds"]["combined"]
        video_targets = self.multi_fold_results[this_fold+3]["total_preds_target"]

        # print(targets_tens.shape, video_targets.shape, audio_targets.shape)
        # if len(targets_tens) == len(video_targets) == len(audio_targets):
        #     print("All targets are the same")
        if len(targets_tens) == len(video_targets) == len(audio_targets) and (targets_tens == video_targets).all() and (video_targets == audio_targets).all():

            predictions = [ audio_preds.argmax(-1) == audio_targets,
                            video_preds.argmax(-1) == video_targets,
                            total_preds == targets_tens]
            cm = create_conf(predictions)
            cm = np.round(cm, 2)
            cue_audio = cm[1, 1] / (cm[1].sum())
            cue_video = cm[2, 1] / (cm[2].sum())
            synergy = cm[0, 1] / (cm[0].sum())
            coexistence = cm[3, 1] / (cm[3].sum())
            return {"cue_audio": cue_audio, "cue_video": cue_video, "synergy":synergy, "coexistence":coexistence}

    def calculate_classification_metrics(self, tts, preds):
        multiclass = True

        ece = torchmetrics.CalibrationError(num_classes=self.config.model.args.num_classes, task="multiclass")
        total_preds, total_preds_nonargmaxed, metrics = {}, {}, defaultdict(dict)
        for pred_key in preds[0].keys():
            total_preds_nonargmaxed[pred_key] = np.concatenate([pred[pred_key].cpu().numpy() for pred in preds], axis=0)
            metrics["loss"][pred_key] = torch.nn.CrossEntropyLoss()(torch.tensor(total_preds_nonargmaxed[pred_key]),
                                                                    torch.tensor(tts)).item()
            metrics["entropy"][pred_key] = entropy(softmax(total_preds_nonargmaxed[pred_key], axis=1), axis=1)
            total_preds[pred_key] = total_preds_nonargmaxed[pred_key].argmax(axis=-1)
            metrics["entropy_correct"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] == tts].mean()
            metrics["entropy_correct_var"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] == tts].std()
            metrics["entropy_wrong"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] != tts].mean()
            metrics["entropy_wrong_var"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] != tts].std()
            metrics["entropy_var"][pred_key] = metrics["entropy"][pred_key].std()
            metrics["entropy"][pred_key] = metrics["entropy"][pred_key].mean()
            metrics["ece_correct"][pred_key] = ece(
                torch.from_numpy(total_preds_nonargmaxed[pred_key][total_preds[pred_key] == tts]),
                torch.from_numpy(tts[total_preds[pred_key] == tts]))
            metrics["ece_false"][pred_key] = ece(
                torch.from_numpy(total_preds_nonargmaxed[pred_key][total_preds[pred_key] != tts]),
                torch.from_numpy(tts[total_preds[pred_key] != tts]))
            metrics["ece"][pred_key] = ece(torch.from_numpy(total_preds_nonargmaxed[pred_key]), torch.from_numpy(tts))
            metrics["acc"][pred_key] = np.equal(tts, total_preds[pred_key]).sum() / len(tts)
            # if self.config.model.args.num_classes >5:
            #     metrics["top5_acc"][pred_key] = top_k_accuracy_score(tts, total_preds_nonargmaxed[pred_key], k=5)
            metrics["f1"][pred_key] = f1_score(total_preds[pred_key], tts, average="macro")
            metrics["k"][pred_key] = cohen_kappa_score(total_preds[pred_key], tts)
            metrics["f1_perclass"][pred_key] = f1_score(total_preds[pred_key], tts, average=None)
            metrics["auc"][pred_key] = roc_auc_score(tts, total_preds[pred_key]) if not multiclass else 0
            metrics["conf"][pred_key] = confusion_matrix(tts, total_preds[pred_key])

            # if pred_key == "combined":
            #     ceu = self.ceu(total_preds[pred_key], tts)
            #     if ceu is not None:
            #         metrics["ceu"][pred_key] = ceu
            #         print(metrics["ceu"][pred_key])

            tp, fp, tn, fn = self._perf_measure(tts, preds)
            metrics["spec"][pred_key] = tn / (tn + fp) if (tn + fp) != 0 else 0
            metrics["sens"][pred_key] = tp / (tp + fn) if (tp + fn) != 0 else 0
        metrics["total_preds"] = total_preds_nonargmaxed
        metrics["total_preds_target"] = tts
        metrics = dict(metrics)  # Avoid passing empty dicts to logs, better return an error!

        return metrics

    def calculate_regression_metrics(self, targets_tens, preds):
        metrics  = defaultdict(dict)
        for pred_key in preds[0].keys():
            total_preds = np.concatenate([pred[pred_key].cpu() for pred in preds], axis=0).squeeze()

            # total_preds = torch.concatenate(total_preds_nonargmaxed[pred_key]).cpu().squeeze()#[:self.processed_instances]

            binary_truth_nozeros = (targets_tens[targets_tens!=0] > 0)
            binary_preds_nozeros = (total_preds[targets_tens!=0] > 0)

            binary_truth = (targets_tens > 0)
            binary_preds = (total_preds > 0)

           #turn them to torch
            binary_truth_nozeros = torch.tensor(binary_truth_nozeros)
            binary_preds_nozeros = torch.tensor(binary_preds_nozeros)

            binary_truth = torch.tensor(binary_truth)
            binary_preds = torch.tensor(binary_preds)

            metrics["acc"][pred_key] = Accuracy(task="binary")(binary_preds_nozeros,binary_truth_nozeros).item()
            metrics["acc_has0"][pred_key] = Accuracy(task="binary")(binary_preds,binary_truth).item()

            metrics["f1"][pred_key] = f1_score(binary_preds_nozeros.cpu().numpy(), binary_truth_nozeros.cpu().numpy(), average='weighted')
            metrics["f1_has0"][pred_key] = f1_score(binary_preds.cpu().numpy(), binary_truth.cpu().numpy(), average='weighted')

            test_preds_a7 = np.clip(total_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(targets_tens, a_min=-3., a_max=3.)
            test_preds_a5 = np.clip(total_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(targets_tens, a_min=-2., a_max=2.)

            metrics["mae"][pred_key] = np.mean(np.absolute(total_preds - targets_tens))  # Average L1 distance between preds and truths
            metrics["corr"][pred_key] = np.corrcoef(total_preds, targets_tens)[0][1]
            metrics["acc_7"][pred_key] = multiclass_acc(test_preds_a7, test_truth_a7)
            metrics["acc_5"][pred_key] = multiclass_acc(test_preds_a5, test_truth_a5)
            if pred_key == "combined":
                metrics["total_preds"] = total_preds
                metrics["total_preds_target"] = targets_tens

        metrics = dict(metrics) #Avoid passing empty dicts to logs, better return an error!

        return metrics
    def validate(self, data_loader, description, show_border_info=False, skip_modality="full"):
            self.model.eval()
            self.model.train(False)
            with torch.no_grad():
                tts, preds, inits = [], [], []
                pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=description, leave=False)
                for batch_idx, served_dict in pbar:

                    if type(served_dict) == tuple:
                        served_dict = {"data": {"c": served_dict[0][0], "f": served_dict[0][1], "g": served_dict[0][2]},
                                       "label": served_dict[3].squeeze(dim=1)}
                        if self.config.get("task", "classification") == "classification" and len(
                                served_dict["label"][served_dict["label"] == -1]) > 0:
                            served_dict["label"][served_dict["label"] == -1] = 0

                    served_dict["data"] = {view: served_dict["data"][view].to(self.device) for view in
                                        served_dict["data"] if type(served_dict["data"][view]) is torch.Tensor}
                    served_dict["data"].update({view: served_dict["data"][view].float() for view in served_dict["data"] if type(view) == int})


                    served_dict["label"] = served_dict["label"].flatten(
                        start_dim=0, end_dim=1).to(self.device)
                    target = served_dict["label"]

                    # target = served_dict["label"][list(served_dict["label"].keys())[0]].flatten(start_dim=0, end_dim=1).to(self.device)
                    # target = served_dict["label"].to(self.device)


                    output = self.model(served_dict["data"])

                    if len(target.shape) > 1:
                        target = target.flatten()

                    tts.append(target)
                    preds.append(output["preds"])
                    pbar.set_description("{} batch {}/{}".format(description, int(batch_idx), int(len(data_loader))))
                    pbar.refresh()


                if "softlabels" in self.config.dataset and self.config.dataset.softlabels:
                    tts = torch.cat(tts).argmax(dim=1).cpu().numpy()
                else:
                    tts = torch.cat(tts).cpu().numpy().squeeze()

                # preds = torch.cat(preds).cpu().numpy()

            if self.config.get("task", "classification") == "classification":
                metrics = self.calculate_classification_metrics(tts, preds)
            elif self.config.get("task", "classification") == "regression":
                metrics = self.calculate_regression_metrics(tts, preds)

            return metrics


    def validate_RF(self, model, data_loader, description, show_border_info=False, skip_modality="full"):
            self.model.eval()
            with torch.no_grad():
                tts, preds, output = [], [], {}
                pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
                # for batch_idx, (data, target, init, _) in pbar:
                for batch_idx, served_dict in pbar:

                    served_dict["data"] = {view: served_dict["data"][view].float().to(self.device) for view in served_dict["data"]}
                    target = served_dict["label"].to(self.device)
                    features = self.model(served_dict["data"])["features"]
                    output["preds"]={"combined": model.predict(torch.concat([features["c"],features["g"]],dim=1).cpu())}

                    data_shuffled_color = copy.deepcopy(served_dict["data"])
                    data_shuffled_color[0] = data_shuffled_color["0_random_indistr"]
                    features = self.model(data_shuffled_color)["features"]
                    output["preds"]["combined_shc"]= model.predict(torch.concat([features["c"],features["g"]],dim=1).cpu())

                    data_shuffled_gray = copy.deepcopy(served_dict["data"])
                    data_shuffled_gray[1] = data_shuffled_color["1_random_indistr"]
                    features = self.model(data_shuffled_gray)["features"]
                    output["preds"]["combined_shg"]= model.predict(torch.concat([features["c"],features["g"]],dim=1).cpu())

                    tts.append(target)
                    preds.append(output["preds"])
                    pbar.set_description("{} batch {}/{}".format(description, int(batch_idx), int(len(data_loader))))
                    pbar.refresh()

                if "softlabels" in self.config.dataset and self.config.dataset.softlabels:
                    tts = torch.cat(tts).argmax(dim=1).cpu().numpy()
                else:
                    tts = torch.cat(tts).cpu().numpy().squeeze()

                # preds = torch.cat(preds).cpu().numpy()

            multiclass = True

            total_preds, total_preds_nonargmaxed, metrics = {}, {}, defaultdict(dict)
            for pred_key in preds[0].keys():
                total_preds[pred_key] = np.concatenate([pred[pred_key] for pred in preds], axis=0)
                # metrics["entropy"][pred_key] = entropy(softmax(total_preds_nonargmaxed[pred_key], axis=1), axis=1)
                # total_preds[pred_key] = total_preds_nonargmaxed[pred_key].argmax(axis=-1)
                # metrics["entropy_correct"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] == tts].mean()
                # metrics["entropy_correct_var"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] == tts].std()
                # metrics["entropy_wrong"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] != tts].mean()
                # metrics["entropy_wrong_var"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] != tts].std()
                # metrics["entropy_var"][pred_key] = metrics["entropy"][pred_key].std()
                # metrics["entropy"][pred_key] = metrics["entropy"][pred_key].mean()
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
            # metrics["total_preds"] = total_preds_nonargmaxed
            metrics["total_preds_target"] = tts
            metrics = dict(metrics)  # Avoid passing empty dicts to logs, better return an error!

            return metrics
    def validate_adv(self, data_loader, description, show_border_info=False, skip_modality="full"):

        # self._get_loss_weights()
        self.w_loss = {"eeg":1,"eog":1,"combined":1, "alignments":0.1, "order":0, "consistency":0, "reconstruction":0}

        self.return_matches= True if self.w_loss["alignments"]!=0 else False
        self.return_order= True if self.w_loss["order"]!=0 else False
        self.return_consistency= True if self.w_loss["consistency"]!=0 else False
        self.return_reconstruction= True if self.w_loss["reconstruction"]!=0 else False

        self.clean_train = self.config.model.args.clean_train if "clean_train" in self.config.model.args else False

        self.alignment_loss = nn.CrossEntropyLoss()
        self.alignment_target = torch.eye(n=500).unsqueeze(dim=0).repeat(500, 1, 1)[
                                :self.config.training_params.batch_size,
                                :self.config.dataset.seq_length[0], :self.config.dataset.seq_length[0]].argmax(dim=-1).cuda()

        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.optimizer.learning_rate,
                                          betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                                          eps=1e-07,
                                          weight_decay=self.config.optimizer.weight_decay)
        def calculate_loss (output, target, skip_modality="full"):
                total_loss =  torch.zeros(1).squeeze().to(self.device)
                output_losses, ce_loss = {}, {}
                if "preds" not in output: output["preds"] = {}

                for k, v in output["preds"].items():
                    if self.w_loss[k]!=0:
                        this_target = target
                        if "incomplete_idx" in output:
                            this_target = this_target[output["incomplete_idx"][k].flatten().bool()]

                        if len(this_target)>0: #TODO: Check if this one needs to be one or zero
                            ce_loss[k] = self.loss(v, this_target)
                            total_loss += self.w_loss[k] * ce_loss[k]
                            ce_loss[k] = ce_loss[k].detach().cpu().numpy()
                            output_losses.update({"ce_loss_{}".format(k): ce_loss[k]})

                if self.return_matches:
                    matches = output["matches"]
                    if matches is not None and type(matches) is dict and "stft_eeg" in matches and matches["stft_eeg"] is not None:
                        if len(matches["stft_eeg"].shape)==2:
                            alignment_target = torch.arange(matches["stft_eeg"].shape[0]).to(matches["stft_eeg"].device)
                        else:
                            alignment_target = torch.arange(matches["stft_eeg"].shape[1]).tile(matches["stft_eeg"].shape[0]).to(matches["stft_eeg"].device)
                            matches["stft_eeg"] = matches["stft_eeg"].flatten(start_dim=0, end_dim=1)
                            matches["stft_eog"] = matches["stft_eog"].flatten(start_dim=0, end_dim=1)

                        alignment_loss = self.alignment_loss(matches["stft_eeg"], alignment_target)
                        alignment_loss += self.alignment_loss(matches["stft_eog"], alignment_target)
                        total_loss += self.w_loss["alignments"]*alignment_loss
                        alignment_loss = alignment_loss.detach().cpu().numpy()
                        output_losses.update({"alignment_loss": alignment_loss})
                        del alignment_loss
                    else:
                        output_losses.update({"alignment_loss": np.array(0, dtype=np.float32)})

                return total_loss, output_losses



        self.model.train()
        tts, preds, inits = [], [], []
        pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
        for batch_idx, served_dict in pbar:

            served_dict["data"] = {view: served_dict["data"][view].float().to(self.device) for view in served_dict["data"]}
            target = served_dict["label"][list(served_dict["label"].keys())[0]].flatten(start_dim=0, end_dim=1).to(self.device)
            for view in served_dict["data"]:
                served_dict["data"][view].requires_grad = True
            self.optimizer.zero_grad()
            pred = self.model(served_dict["data"], return_matches=self.return_matches)
            total_loss, output_losses = calculate_loss(pred, target)
            total_loss.backward(retain_graph=True)
            data_plus = {view: served_dict["data"][view] + self.config.training_params.adversarial_training.adv_epsilon * (served_dict["data"][view].grad).sign() for view in served_dict["data"]}
            self.optimizer.zero_grad()
            self.model.eval()
            with torch.no_grad():
                preds_plus = self.model(data_plus, return_matches=self.return_matches)

            # for view in served_dict["data"]:
            #     print(torch.equal(data_plus[view], served_dict["data"][view]))
            # print(preds_plus["preds"]["combined"][0])
            # print(pred["preds"]["combined"][0])
            # print(torch.equal(preds_plus["preds"]["combined"], pred["preds"]["combined"]))
            # print(torch.equal(preds_plus["preds"]["eeg"], pred["preds"]["eeg"]))
            # print(torch.equal(preds_plus["preds"]["eog"], pred["preds"]["eog"]))

            del data_plus, total_loss, output_losses
            tts.append(target.detach().cpu())

            preds.append(preds_plus["preds"])
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
            total_preds[pred_key] = np.concatenate([pred[pred_key].detach().cpu().numpy() for pred in preds], axis=0)
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
        metrics = dict(metrics)  # Avoid passing empty dicts to logs, better return an error!

        return metrics
    def attend(self, data_loader, description, show_border_info=False, skip_modality="full"):
            self.model.eval()
            with torch.no_grad():
                tts, preds, inits = [], [], []
                ca_attentions = {"eeg":{"inner": {"layer_0":[], "layer_1":[], "layer_2":[], "layer_3":[]},
                                        "outer":{"layer_0": [], "layer_1": [], "layer_2": [], "layer_3": []}},
                                 "eog": {"inner": {"layer_0": [], "layer_1": [], "layer_2": [], "layer_3": []},
                                         "outer": {"layer_0": [], "layer_1": [], "layer_2": [], "layer_3": []}}}
                pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
                # for batch_idx, (data, target, init, _) in pbar:
                for batch_idx, served_dict in pbar:

                    served_dict["data"] = {view: served_dict["data"][view].float().to(self.device) for view in served_dict["data"]}
                    target = served_dict["label"][list(served_dict["label"].keys())[0]].flatten(start_dim=0, end_dim=1).to(self.device)

                    forward_hook_manager = ForwardHookManager(self.device)
                    forward_hook_manager.add_hook(self.model,"enc_0.outer_tf_eeg.tf.layers.0.CA.scaled_dotproduct_attention", requires_input=False, requires_output=True)
                    forward_hook_manager.add_hook(self.model,"enc_0.outer_tf_eeg.tf.layers.1.CA.scaled_dotproduct_attention",requires_input=False, requires_output=True)
                    forward_hook_manager.add_hook(self.model,"enc_0.outer_tf_eeg.tf.layers.2.CA.scaled_dotproduct_attention",requires_input=False, requires_output=True)
                    forward_hook_manager.add_hook(self.model, "enc_0.outer_tf_eeg.tf.layers.3.CA.scaled_dotproduct_attention", requires_input=False, requires_output=True)

                    forward_hook_manager.add_hook(self.model,"enc_0.inner_tf_eeg.tf.layers.0.CA.scaled_dotproduct_attention", requires_input=False, requires_output=True)
                    forward_hook_manager.add_hook(self.model,"enc_0.inner_tf_eeg.tf.layers.1.CA.scaled_dotproduct_attention",requires_input=False, requires_output=True)
                    forward_hook_manager.add_hook(self.model,"enc_0.inner_tf_eeg.tf.layers.2.CA.scaled_dotproduct_attention",requires_input=False, requires_output=True)
                    forward_hook_manager.add_hook(self.model, "enc_0.inner_tf_eeg.tf.layers.3.CA.scaled_dotproduct_attention", requires_input=False, requires_output=True)


                    forward_hook_manager.add_hook(self.model,"enc_0.outer_tf_eog.tf.layers.3.CA.scaled_dotproduct_attention", requires_input=False, requires_output=True)
                    forward_hook_manager.add_hook(self.model,"enc_0.outer_tf_eog.tf.layers.3.CA.scaled_dotproduct_attention",
                                                  requires_input=False, requires_output=True)
                    forward_hook_manager.add_hook(self.model,
                                                  "enc_0.outer_tf_eog.tf.layers.3.CA.scaled_dotproduct_attention",
                                                  requires_input=False, requires_output=True)
                    forward_hook_manager.add_hook(self.model,
                                                  "enc_0.outer_tf_eog.tf.layers.3.CA.scaled_dotproduct_attention",
                                                  requires_input=False, requires_output=True)


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

                    io_dict = forward_hook_manager.pop_io_dict()

                    ca_attentions["eeg"]["outer"]["layer_0"].append( io_dict["enc_0.outer_tf_eeg.tf.layers.0.CA.scaled_dotproduct_attention"]["output"][1].cpu())
                    ca_attentions["eeg"]["outer"]["layer_1"].append( io_dict["enc_0.outer_tf_eeg.tf.layers.1.CA.scaled_dotproduct_attention"]["output"][1].cpu())
                    ca_attentions["eeg"]["outer"]["layer_2"].append( io_dict["enc_0.outer_tf_eeg.tf.layers.2.CA.scaled_dotproduct_attention"]["output"][1].cpu())
                    ca_attentions["eeg"]["outer"]["layer_3"].append( io_dict["enc_0.outer_tf_eeg.tf.layers.3.CA.scaled_dotproduct_attention"]["output"][1].cpu())

                    ca_attentions["eeg"]["inner"]["layer_0"].append( io_dict["enc_0.inner_tf_eeg.tf.layers.0.CA.scaled_dotproduct_attention"]["output"][1].cpu())
                    ca_attentions["eeg"]["inner"]["layer_1"].append( io_dict["enc_0.inner_tf_eeg.tf.layers.1.CA.scaled_dotproduct_attention"]["output"][1].cpu())
                    ca_attentions["eeg"]["inner"]["layer_2"].append( io_dict["enc_0.inner_tf_eeg.tf.layers.2.CA.scaled_dotproduct_attention"]["output"][1].cpu())
                    ca_attentions["eeg"]["inner"]["layer_3"].append( io_dict["enc_0.inner_tf_eeg.tf.layers.3.CA.scaled_dotproduct_attention"]["output"][1].cpu())



                    tts.append(target)
                    preds.append(pred["preds"])
                    pbar.set_description("{} batch {}/{}".format(description, int(batch_idx), int(len(data_loader))))
                    pbar.refresh()
            plt.figure(figsize=(24, 6))
            for i in range(4):
                plt.subplot(int("14{}".format(i + 1)))
                ca_attentions["eeg"]["outer"]["layer_{}".format(i)] = torch.cat(ca_attentions["eeg"]["outer"]["layer_{}".format(i)], dim=0)
                ca_attentions["eeg"]["inner"]["layer_{}".format(i)] = torch.cat(ca_attentions["eeg"]["inner"]["layer_{}".format(i)], dim=0)
                sns.heatmap(ca_attentions["eeg"]["inner"]["layer_{}".format(i)].mean(dim=0).numpy()[1:,1:], cbar=False)
                plt.title("Inner Layer {}, norm {}".format(i, np.linalg.norm(ca_attentions["eeg"]["inner"]["layer_{}".format(i)].numpy()[1:,1:])))
                # sns.heatmap(ca_attentions["eeg"]["outer"]["layer_{}".format(i)].mean(dim=0).numpy(), cbar=False)
                # plt.title("Outer Layer {}".format(i))
                # plt.show()
            plt.tight_layout()
            plt.show()

            return 0
    # def cca(self, data_loader, description, show_border_info=False, skip_modality="full"):
    #     cca_loss = CCA_Loss(128, True, "cpu")
    #     self.model.eval()
    #     with torch.no_grad():
    #         tts, preds, inits = [], [], []
    #         ca_attentions = {"eeg":{"inner": {"layer_0":[], "layer_1":[], "layer_2":[], "layer_3":[]},
    #                                 "outer":{"layer_0": [], "layer_1": [], "layer_2": [], "layer_3": []}},
    #                          "eog": {"inner": {"layer_0": [], "layer_1": [], "layer_2": [], "layer_3": []},
    #                                  "outer": {"layer_0": [], "layer_1": [], "layer_2": [], "layer_3": []}}}
    #         inner_cls_features = {"eeg": [], "eog": []}
    #         pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    #         # for batch_idx, (data, target, init, _) in pbar:
    #         for batch_idx, served_dict in pbar:
    #
    #             served_dict["data"] = {view: served_dict["data"][view].float().to(self.device) for view in served_dict["data"]}
    #             target = served_dict["label"][list(served_dict["label"].keys())[0]].flatten(start_dim=0, end_dim=1).to(self.device)
    #
    #             forward_hook_manager = ForwardHookManager(self.device)
    #             # forward_hook_manager.add_hook(self.model,"enc_0.outer_tf_eeg.tf.layers.0.CA.scaled_dotproduct_attention", requires_input=False, requires_output=True)
    #             # forward_hook_manager.add_hook(self.model,"enc_0.outer_tf_eeg.tf.layers.1.CA.scaled_dotproduct_attention",requires_input=False, requires_output=True)
    #             # forward_hook_manager.add_hook(self.model,"enc_0.outer_tf_eeg.tf.layers.2.CA.scaled_dotproduct_attention",requires_input=False, requires_output=True)
    #             # forward_hook_manager.add_hook(self.model, "enc_0.outer_tf_eeg.tf.layers.3.CA.scaled_dotproduct_attention", requires_input=False, requires_output=True)
    #             #
    #             # forward_hook_manager.add_hook(self.model,"enc_0.inner_tf_eeg.tf.layers.0.CA.scaled_dotproduct_attention", requires_input=False, requires_output=True)
    #             # forward_hook_manager.add_hook(self.model,"enc_0.inner_tf_eeg.tf.layers.1.CA.scaled_dotproduct_attention",requires_input=False, requires_output=True)
    #             # forward_hook_manager.add_hook(self.model,"enc_0.inner_tf_eeg.tf.layers.2.CA.scaled_dotproduct_attention",requires_input=False, requires_output=True)
    #             # forward_hook_manager.add_hook(self.model, "enc_0.inner_tf_eeg.tf.layers.3.CA.scaled_dotproduct_attention", requires_input=False, requires_output=True)
    #             #
    #             # forward_hook_manager.add_hook(self.model,"enc_0.outer_tf_eog.tf.layers.0.CA.scaled_dotproduct_attention", requires_input=False, requires_output=True)
    #             # forward_hook_manager.add_hook(self.model,"enc_0.outer_tf_eog.tf.layers.1.CA.scaled_dotproduct_attention",requires_input=False, requires_output=True)
    #             # forward_hook_manager.add_hook(self.model,"enc_0.outer_tf_eog.tf.layers.2.CA.scaled_dotproduct_attention",requires_input=False, requires_output=True)
    #             # forward_hook_manager.add_hook(self.model, "enc_0.outer_tf_eog.tf.layers.3.CA.scaled_dotproduct_attention", requires_input=False, requires_output=True)
    #             #
    #             # forward_hook_manager.add_hook(self.model,"enc_0.inner_tf_eog.tf.layers.0.CA.scaled_dotproduct_attention", requires_input=False, requires_output=True)
    #             # forward_hook_manager.add_hook(self.model,"enc_0.inner_tf_eog.tf.layers.1.CA.scaled_dotproduct_attention",requires_input=False, requires_output=True)
    #             # forward_hook_manager.add_hook(self.model,"enc_0.inner_tf_eog.tf.layers.2.CA.scaled_dotproduct_attention",requires_input=False, requires_output=True)
    #             # forward_hook_manager.add_hook(self.model, "enc_0.inner_tf_eog.tf.layers.3.CA.scaled_dotproduct_attention", requires_input=False, requires_output=True)
    #
    #             #Late-CoRe
    #             # forward_hook_manager.add_hook(self.model, "enc_0.inner_tf_eeg.tf.layers.3", requires_input=False, requires_output=True)
    #             # forward_hook_manager.add_hook(self.model, "enc_0.inner_tf_eog.tf.layers.3", requires_input=False, requires_output=True)
    #
    #             #Early
    #             forward_hook_manager.add_hook(self.model, "enc_0.inner_tf.tf.layers.3", requires_input=False, requires_output=True)
    #             forward_hook_manager.add_hook(self.model, "enc_0.inner_tf.tf.layers.3", requires_input=False, requires_output=True)
    #
    #             pred = self.model(served_dict["data"])
    #
    #             io_dict = forward_hook_manager.pop_io_dict()
    #
    #             # ca_attentions["eeg"]["outer"]["layer_0"].append( io_dict["enc_0.outer_tf_eeg.tf.layers.0.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             # ca_attentions["eeg"]["outer"]["layer_1"].append( io_dict["enc_0.outer_tf_eeg.tf.layers.1.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             # ca_attentions["eeg"]["outer"]["layer_2"].append( io_dict["enc_0.outer_tf_eeg.tf.layers.2.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             # ca_attentions["eeg"]["outer"]["layer_3"].append( io_dict["enc_0.outer_tf_eeg.tf.layers.3.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             #
    #             # ca_attentions["eeg"]["inner"]["layer_0"].append( io_dict["enc_0.inner_tf_eeg.tf.layers.0.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             # ca_attentions["eeg"]["inner"]["layer_1"].append( io_dict["enc_0.inner_tf_eeg.tf.layers.1.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             # ca_attentions["eeg"]["inner"]["layer_2"].append( io_dict["enc_0.inner_tf_eeg.tf.layers.2.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             # ca_attentions["eeg"]["inner"]["layer_3"].append( io_dict["enc_0.inner_tf_eeg.tf.layers.3.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             #
    #             # ca_attentions["eog"]["outer"]["layer_0"].append( io_dict["enc_0.inner_tf_eog.tf.layers.0.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             # ca_attentions["eog"]["outer"]["layer_1"].append( io_dict["enc_0.inner_tf_eog.tf.layers.1.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             # ca_attentions["eog"]["outer"]["layer_2"].append( io_dict["enc_0.inner_tf_eog.tf.layers.2.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             # ca_attentions["eog"]["outer"]["layer_3"].append( io_dict["enc_0.inner_tf_eog.tf.layers.3.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             #
    #             # ca_attentions["eog"]["inner"]["layer_0"].append( io_dict["enc_0.inner_tf_eog.tf.layers.0.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             # ca_attentions["eog"]["inner"]["layer_1"].append( io_dict["enc_0.inner_tf_eog.tf.layers.1.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             # ca_attentions["eog"]["inner"]["layer_2"].append( io_dict["enc_0.inner_tf_eog.tf.layers.2.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #             # ca_attentions["eog"]["inner"]["layer_3"].append( io_dict["enc_0.inner_tf_eog.tf.layers.3.CA.scaled_dotproduct_attention"]["output"][1].cpu())
    #
    #             #Early
    #             inner_cls_features["eeg"].append(io_dict["enc_0.inner_tf.tf.layers.3"]["output"][0, :int(io_dict["enc_0.inner_tf.tf.layers.3"]["output"].shape[1]/2), :].cpu())
    #             inner_cls_features["eog"].append(io_dict["enc_0.inner_tf.tf.layers.3"]["output"][0, int(io_dict["enc_0.inner_tf.tf.layers.3"]["output"].shape[1]/2):, :].squeeze().cpu())
    #
    #             #Late-CoRe
    #             # inner_cls_features["eeg"].append(io_dict["enc_0.inner_tf_eeg.tf.layers.3"]["output"][0].cpu())
    #             # inner_cls_features["eog"].append(io_dict["enc_0.inner_tf_eog.tf.layers.3"]["output"][0].cpu())
    #
    #             tts.append(target)
    #             preds.append(pred["preds"])
    #             pbar.set_description("{} batch {}/{}".format(description, int(batch_idx), int(len(data_loader))))
    #             pbar.refresh()
    #
    #     inner_cls_features["eeg"] = torch.cat(inner_cls_features["eeg"], dim=0)
    #     inner_cls_features["eog"] = torch.cat(inner_cls_features["eog"], dim=0)
    #     correlation = cca_loss.loss(inner_cls_features["eeg"], inner_cls_features["eog"])
    #     print("CCA correlation: {}".format(correlation))
    #
    #     return correlation

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
            # message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i, metrics["k"][i])
            # latex_message += " {:.3f} &".format(metrics["k"][i])
            # message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i, metrics["f1"][i] * 100)
            # latex_message += " {:.1f} &".format(metrics["f1"][i] * 100)
            # message += Fore.LIGHTRED_EX + "ECE_{}: {:.3f} ".format(i, metrics["ece"][i])
            # latex_message += " {:.3f} &".format(metrics["ece"][i])
            # message += Fore.BLUE + "F1_perclass_{}: {} ".format(i,"{}".format(str(list((metrics["f1_perclass"][i] * 100).round(2)))))
            # for i in list((metrics["f1_perclass"][i] * 100).round(2)):
            #     latex_message += " {:.1f} &".format(i)
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

    def update_ece_test_results(self, checkpoint, save_dir, test_results, skipped=False):

        checkpoint["post_test_results"]["ece"] = test_results["ece"]
        try:
            torch.save(checkpoint, save_dir)
            print("Models has saved successfully in {}".format(save_dir))
        except:
            raise Exception("Problem in model saving")
    def save_test_results_adv(self, checkpoint, save_dir, test_results, skipped=False):
        if "post_test_results_adv" not in checkpoint:
            checkpoint["post_test_results_adv"] = {}
        checkpoint["post_test_results_adv"][self.config.training_params.adversarial_training.adv_epsilon] =  test_results
        try:
            torch.save(checkpoint, save_dir)
            print("Models has saved successfully in {}".format(save_dir))
        except:
            raise Exception("Problem in model saving")
    def save_test_results_rebase(self, checkpoint, save_dir):

        test_results_dict = { "post_test_results_adv": {self.config.training_params.adversarial_training.adv_epsilon: checkpoint["post_test_results_adv"]}}
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
        # ax.xaxis.set_ticklabels(['Wake', 'N1','N2', 'N3', 'REM'])
        # ax.yaxis.set_ticklabels(['REM', 'N3','N2', 'N1', 'Wake'])
        plt.show()

