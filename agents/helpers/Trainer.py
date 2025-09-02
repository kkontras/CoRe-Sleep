import torch
import time
from tqdm import tqdm
from collections import defaultdict
from colorama import Fore
from graphs.models.SHHS_Models_new import *

class Trainer():

    def __init__(self, agent):
        self.agent = agent
        self._get_loss_weights()
        self.validate_every = self.agent.config.early_stopping.validate_every
        self.validate_after = self.agent.config.early_stopping.validate_after
        self.end_of_epoch_check = self.agent.config.early_stopping.get("end_of_epoch_check", False)
        if self.end_of_epoch_check:
            self.agent.config.early_stopping.validate_every = len(self.agent.data_loader.train_loader)


    def sleep_train(self):

        self.agent.model.train()
        self.agent.start = time.time()

        for self.agent.logs["current_epoch"] in range(self.agent.logs["current_epoch"], self.agent.config.early_stopping.max_epoch):
            self.agent.evaluators.train_evaluator.reset()
            pbar = tqdm(enumerate(self.agent.data_loader.train_loader), desc="Training", leave=None, disable=self.agent.config.training_params.tdqm_disable, position=0, total=len(self.agent.data_loader.train_loader))
            for batch_idx, served_dict in pbar:

                self.agent.optimizer.zero_grad()

                self.sleep_train_one_step(served_dict)

                self.agent.optimizer.step()
                self.agent.scheduler.step(self.agent.logs["current_step"]+1)

                del served_dict

                pbar_message = self.local_logging(batch_idx, False)
                pbar.set_description(pbar_message)
                pbar.refresh()

                if self.agent.evaluators.train_evaluator.get_early_stop(): return

                self.agent.logs["current_step"] += 1

    def local_logging(self, batch_idx, end_of_epoch=None):

        mean_batch_loss, mean_batch_loss_message = self.agent.evaluators.train_evaluator.mean_batch_loss()

        if self.end_of_epoch_check and end_of_epoch or not self.end_of_epoch_check and self.agent.logs["current_step"] % self.agent.config.early_stopping.validate_every == 0 and \
                    self.agent.logs["current_step"] // self.agent.config.early_stopping.validate_every >= self.agent.config.early_stopping.validate_after and \
                    batch_idx != 0:

            self.agent.validator_tester.validate()
            if self.agent.config.training_params.rec_test:
                self.agent.validator_tester.validate(test_set=True)
            self.agent.monitor_n_saver.monitoring()
            if self.agent.evaluators.train_evaluator.get_early_stop(): return
            self.agent.model.train()


        pbar_message = Fore.WHITE + "Training batch {0:d}/{1:d} steps no improve {2:d} with {3:}".format(batch_idx,
                                                                                                     len(self.agent.data_loader.train_loader) - 1,
                                                                                                     self.agent.logs["steps_no_improve"], mean_batch_loss_message)
        # pbar_message += " time {:.1f} sec/step, ".format(self.running_values["prev_epoch_time"])
        # pbar_message += "saved at {}".format(self.running_values["saved_at_valstep"])
        return pbar_message


    def _calc_superv_loss(self, pred_key, preds, target, output, output_losses, total_loss):

        if self.w_loss[pred_key] != 0:
            this_target = target
            if "incomplete_idx" in output:
                this_target = this_target[output["incomplete_idx"][pred_key].flatten().bool()]

            if len(this_target) > 0:
                ce_loss = self.agent.loss(preds, this_target)
                total_loss += self.w_loss[pred_key] * ce_loss
                output_losses.update({"ce_loss_{}".format(pred_key): ce_loss.detach().cpu()})
        return total_loss, output_losses

    # def _calc_alignment_loss(self, output, output_losses, total_loss):
    #     matches = output["matches"]
    #     if matches is not None and isinstance(matches, dict) and "stft_eeg" in matches and matches[
    #         "stft_eeg"] is not None:
    #         if len(matches["stft_eeg"].shape) == 2:
    #             alignment_target = torch.arange(matches["stft_eeg"].shape[0]).cuda()
    #         else:
    #             alignment_target = torch.arange(matches["stft_eeg"].shape[1]).tile(matches["stft_eeg"].shape[0]).cuda()
    #             matches["stft_eeg"] = matches["stft_eeg"].flatten(start_dim=0, end_dim=1)
    #             matches["stft_eog"] = matches["stft_eog"].flatten(start_dim=0, end_dim=1)
    #
    #         alignment_loss = self.agent.alignment_loss(matches["stft_eeg"], alignment_target)
    #         alignment_loss += self.agent.alignment_loss(matches["stft_eog"], alignment_target)
    #         total_loss += self.w_loss["alignments"] * alignment_loss
    #         alignment_loss = alignment_loss.detach().cpu()
    #         output_losses.update({"alignment_loss": alignment_loss})
    #         del alignment_loss
    #     # else:
    #     #     output_losses.update({"alignment_loss": np.array(0, dtype=np.float32)})


    def sleep_train_one_step(self, served_dict):

        return_matches = self.w_loss["alignments"] != 0

        served_dict["data"] = {view: served_dict["data"][view].float().to(self.agent.device) for view in served_dict["data"]}
        # print(served_dict["label"].shape)
        served_dict["label"] = served_dict["label"].flatten(start_dim=0, end_dim=1).to(self.agent.device)
        del served_dict["skip_view"]

        target = served_dict["label"]

        output = self.get_predictions_time_series_onlyskip(served_dict=served_dict,
                                                           return_matches=return_matches)


        output_losses, total_loss = {}, torch.Tensor([0]).cuda()

        for pred_key, preds in output["preds"].items():
            self._calc_superv_loss(pred_key, preds, target, output, output_losses, total_loss)

        if "losses" in output:
            total_loss += torch.cat([output["losses"][i].unsqueeze(dim=0) for i in output["losses"]], dim=0).sum()
            output_losses.update({i: output["losses"][i].detach().cpu() for i in output["losses"]})

        if total_loss.requires_grad:
            total_loss.backward()
        else:
            raise ValueError("Here we have a problem, the total loss is empty or does not require gradient, output_losses: {}".format(output_losses))

        total_loss = total_loss.detach().cpu()
        output_losses.update({"total": total_loss})
        del total_loss

        for i in output["preds"]:  output["preds"][i] = output["preds"][i].detach().cpu()

        if "incomplete_idx" not in output: output["incomplete_idx"] = None

        #TODO: Fix evaluator to get all these and especially the incomplete_idx
        self.agent.evaluators.train_evaluator.process(output_losses, output["preds"], target, output["incomplete_idx"])

        return


    def _get_loss_weights(self):
        multi_loss_w = self.agent.config.model.args.get("multi_loss", {})
        if not hasattr(self.agent.logs,"w_loss"):
            w_loss = defaultdict(int)
            supervised_w = multi_loss_w.get("supervised_losses", {})
            for k, v in supervised_w.items():
                w_loss[k] = v
            w_loss["alignments"] = multi_loss_w.get("alignment_loss", 0)
            self.w_loss = w_loss
            self.agent.logs["w_loss"] = w_loss
        elif hasattr(self.agent.logs,"w_loss") and self.agent.config.model.load_ongoing:
            self.w_loss = self.agent.logs.w_loss
        print("Loss Weights are", dict(self.w_loss))


    def get_predictions_time_series_onlyskip(self, served_dict, **kwargs):

        """
        This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
        :param views: List of tensors, data views/modalities
        :param inits: Tensor indicating with value one, when there incontinuities.
        :return: predictions of the self.model on the batch
        """

        #TODO: Comment this method and put proper names on the variables

        views = served_dict["data"]

        this_view = views[list(views.keys())[0]]

        this_skip_modality = torch.zeros(this_view.shape[0], this_view.shape[1])
        if "skip_view" in served_dict and type(served_dict["skip_view"])==dict:
            this_skip_modality[served_dict["skip_view"]["stft_eeg"] == 1] += 1
            this_skip_modality[served_dict["skip_view"]["stft_eog"] == 1] += 2
            this_skip_modality[this_skip_modality==3]=0
        elif "skip_view" in served_dict and served_dict["skip_view"]=="eeg":
            this_skip_modality +=1
        elif "skip_view" in served_dict and served_dict["skip_view"]=="eog":
            this_skip_modality +=2

        batch, outer = this_view.shape[0], this_view.shape[1]

        global_pred = {"preds": {"combined": [], "eeg":[], "eog": []},
                       "matches": {"stft_eeg":[], "stft_eog": []},
                       "incomplete_idx": {"eeg": [], "eog": [], "combined": []}}

        consec_batch_changes, consec_batch_counts = torch.unique_consecutive(this_skip_modality.sum(dim=1), return_counts=True)

        if len(consec_batch_changes)>1:
            b_count = 0
            for b_cons in consec_batch_counts:
                consec_changes, consec_counts = torch.unique_consecutive(this_skip_modality[b_count:b_cons+b_count], return_counts=True)
                if len(consec_counts)>1:
                    for b_cons_i in range(1,b_cons+1):
                        consec_changes, consec_counts = torch.unique_consecutive(this_skip_modality[b_count:1 + b_count], return_counts=True)
                        count = 0
                        local_pred = { "preds": {"combined": [], "eeg": [],  "eog": []},
                                       "matches": {"stft_eeg": [], "stft_eog": []},
                                       "incomplete_idx": {"eeg": [], "eog": [], "combined": []}}

                        for cons in consec_counts:
                            dt_skip = {view: skip_modality[view][b_count:1+b_count, count:cons+count] for view in views} if skip_modality else None
                            pred_split = self.agent.model(
                                {view: views[view][b_count:1+b_count, count:cons+count] for view in views}, skip_modality=dt_skip, **kwargs)

                            for pred in local_pred["preds"]:
                                if pred in pred_split["preds"] and pred in pred_split["preds"]:
                                    local_pred["preds"][pred].append(pred_split["preds"][pred])

                            if "matches" in pred_split and type(pred_split["matches"])==dict:
                                local_pred["matches"]["stft_eeg"].append(pred_split["matches"]["stft_eeg"])
                                local_pred["matches"]["stft_eog"].append(pred_split["matches"]["stft_eog"])

                            for pred in local_pred["incomplete_idx"]:
                                if "incomplete_idx" in pred_split and pred in pred_split["incomplete_idx"]:
                                    local_pred["incomplete_idx"][pred].append(pred_split["incomplete_idx"][pred])

                            count += cons
                        dim_to_concat = {"incomplete_idx":1, "preds":0}
                        for el in ["incomplete_idx", "preds"]:
                            for pred in local_pred[el]:
                                if type(local_pred[el][pred]) == list and local_pred[el][pred]!=[]:
                                    local_pred[el][pred] = torch.cat(local_pred[el][pred], dim=dim_to_concat[el])
                                    global_pred[el][pred].append(local_pred[el][pred])
                        for pred in local_pred["matches"]:
                            if type(local_pred["matches"][pred]) == list:
                                match_init = torch.zeros(1, outer, outer).to(this_view.device)
                                match_count = 0
                                for i in local_pred["matches"][pred]:
                                    if i is not None:
                                        match_init[0,match_count:match_count+i.shape[1],match_count:match_count+i.shape[2]] = i
                                local_pred["matches"][pred] = match_init
                                global_pred["matches"][pred].append(local_pred["matches"][pred])
                        b_count += 1
                else:
                    dt_skip = {view: skip_modality[view][b_count:b_cons+b_count] for view in views} if skip_modality else None
                    pred_split = self.agent.model(
                        {view: views[view][b_count:b_cons+b_count] for view in views}, skip_modality=dt_skip, **kwargs)

                    for pred in global_pred["preds"]:
                        if pred in pred_split["preds"] and pred in pred_split["preds"]:
                            global_pred["preds"][pred].append(pred_split["preds"][pred])

                    if "matches" in pred_split and type(pred_split["matches"])==dict:
                        if pred_split["matches"]["stft_eeg"] is not None:
                            global_pred["matches"]["stft_eeg"].append(pred_split["matches"]["stft_eeg"])
                        if pred_split["matches"]["stft_eog"] is not None:
                            global_pred["matches"]["stft_eog"].append(pred_split["matches"]["stft_eog"])

                    for pred in global_pred["incomplete_idx"]:
                        if "incomplete_idx" in pred_split and pred in pred_split["incomplete_idx"]:
                            global_pred["incomplete_idx"][pred].append(pred_split["incomplete_idx"][pred])
                    b_count += b_cons
            remove_later_list = []
            for el in global_pred:
                for pred in global_pred[el]:
                    if type(global_pred[el][pred]) == list:
                        if len(global_pred[el][pred])>0:
                            global_pred[el][pred] = torch.cat(global_pred[el][pred])
                        else:
                            remove_later_list.append({"el":el, "pred":pred})
            for i in remove_later_list:
                global_pred[i["el"]].pop(i["pred"], None)
        else:
            global_pred = self.agent.model(views, skip_modality=served_dict.get("skip_view", None), **kwargs)

        return global_pred