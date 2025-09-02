import torch
import time

import numpy as np
from tqdm import tqdm
import einops
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix

from collections import defaultdict
from utils.config import process_config
from graphs.models.SHHS_Models_new import *

class Validator_Tester():
    def __init__(self, agent):
        self.agent = agent
        self.multi_supervised = False
        # self.valtest_step_func = self._find_valtest_step_func()
        # self.this_valtest_step_func = getattr(self, self.valtest_step_func)
        self._get_loss_weights()

    def validate(self, best_model = False, test_set= False):
        """
        One cycle of model validation
        :return:
        """
        if best_model:
            self.agent.best_model.eval()
            self.agent.best_model.train(False)

        self.agent.model.eval()
        self.agent.model.train(False)

        this_evaluator = self.agent.evaluators.test_evaluator if test_set else self.agent.evaluators.val_evaluator
        this_dataloader = self.agent.data_loader.test_loader if test_set else self.agent.data_loader.valid_loader
        this_evaluator.reset()
        with torch.no_grad():
            pbar = tqdm(enumerate(this_dataloader),
                        total=len(this_dataloader),
                        desc="Validation",
                        leave=False,
                        disable=not best_model, position=1)
            for batch_idx, served_dict in pbar:

                self.sleep_valtest_one_step(served_dict, this_evaluator, best_model=best_model)

                mean_batch_loss, mean_batch_loss_message = this_evaluator.mean_batch_loss()

                pbar_message = "Validation batch {0:d}/{1:d} with {2:}".format(batch_idx,
                                                                             len(this_dataloader) - 1,
                                                                             mean_batch_loss_message)
                pbar.set_description(pbar_message)
                pbar.refresh()


    def _calc_mean_batch_loss(self, batch_loss):
        mean_batch = defaultdict(list)
        for b_i in batch_loss:
            for loss_key in b_i:
                mean_batch[loss_key].append(b_i[loss_key])
        for key in mean_batch:
            mean_batch[key] = np.array(mean_batch[key]).mean(axis=0)
        return mean_batch

    def _get_loss_weights(self):
        multi_loss_w = self.agent.config.model.args.get("multi_loss", {})
        if not hasattr(self.agent.logs, "w_loss"):
            w_loss = defaultdict(int)
            supervised_w = multi_loss_w.get("supervised_losses", {})
            for k, v in supervised_w.items():
                w_loss[k] = v
            w_loss["alignments"] = multi_loss_w.get("alignment_loss", 0)
            self.w_loss = w_loss
            self.agent.logs["w_loss"] = w_loss
        elif hasattr(self.agent.logs, "w_loss") and self.agent.config.model.load_ongoing:
            self.w_loss = self.agent.logs.w_loss
        self.w_loss["total"] = 1

    def _calc_alignment_loss(self, output, output_losses, total_loss):
        matches = output["matches"]
        if matches is not None and isinstance(matches, dict) and "stft_eeg" in matches and matches[
            "stft_eeg"] is not None:
            if len(matches["stft_eeg"].shape) == 2:
                alignment_target = torch.arange(matches["stft_eeg"].shape[0]).cuda()
            else:
                alignment_target = torch.arange(matches["stft_eeg"].shape[1]).tile(matches["stft_eeg"].shape[0]).cuda()
                matches["stft_eeg"] = matches["stft_eeg"].flatten(start_dim=0, end_dim=1)
                matches["stft_eog"] = matches["stft_eog"].flatten(start_dim=0, end_dim=1)

            alignment_loss = self.agent.alignment_loss(matches["stft_eeg"], alignment_target)
            alignment_loss += self.agent.alignment_loss(matches["stft_eog"], alignment_target)
            total_loss += self.w_loss["alignments"] * alignment_loss
            alignment_loss = alignment_loss.detach().cpu()
            output_losses.update({"alignment_loss": alignment_loss})
            del alignment_loss
        else:
            output_losses.update({"alignment_loss": np.array(0, dtype=np.float32)})

    def sleep_valtest_one_step(self, served_dict, this_evaluator, best_model=False):

            served_dict["data"] = {view: served_dict["data"][view].float().cuda() for view in served_dict["data"]}
            served_dict["label"] = served_dict["label"].flatten(
                start_dim=0, end_dim=1).to(self.agent.device)

            self._get_loss_weights()

            return_matches= True if self.w_loss["alignments"]!=0 else False


            if best_model:
                # output = self.agent.best_model(served_dict["data"], skip_modality=served_dict["skip_view"], inits=served_dict["init"], return_matches=return_matches)
                output = self.agent.best_model(served_dict["data"], return_matches=return_matches)
            else:
                # output = self.agent.model(served_dict["data"], skip_modality=served_dict["skip_view"], inits=served_dict["init"], return_matches=return_matches)
                output = self.agent.model(served_dict["data"], return_matches=return_matches)

            target = served_dict["label"]
            ce_loss = {}
            for k, v in output.get("preds", {}).items():
                ce_loss[k] = self.agent.loss(v, target)

            total_loss = 0
            output_losses = {}
            for i in ce_loss:
                total_loss += self.w_loss[i] * ce_loss[i]
                ce_loss[i] = ce_loss[i].detach().cpu()
                output_losses.update({"ce_loss_{}".format(i): ce_loss[i]})

            if return_matches:
                self._calc_alignment_loss(output, output_losses, total_loss)

            total_loss =  total_loss.detach().cpu()
            output_losses.update({"total": total_loss})

            for i in output["preds"]:  output["preds"][i] =  output["preds"][i].detach().cpu()

            this_evaluator.process(output_losses, output["preds"], target)

    def get_predictions_time_series_onlyskip(self, best_model, views, inits, skip_modality=None, **kwargs):

        """
        This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
        :param views: List of tensors, data views/modalities
        :param inits: Tensor indicating with value one, when there incontinuities.
        :return: predictions of the self.model on the batch
        """

        this_view = views[list(inits.keys())[0]]

        this_skip_modality = torch.zeros(this_view.shape[0], this_view.shape[1])
        if type(skip_modality)==dict:
            this_skip_modality[skip_modality["stft_eeg"] == 1] += 1
            this_skip_modality[skip_modality["stft_eog"] == 1] += 2
            this_skip_modality[this_skip_modality==3]=0
        elif skip_modality=="eeg":
            this_skip_modality +=1
        elif skip_modality=="eog":
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
            global_pred = self.agent.model(views, skip_modality=skip_modality, **kwargs)

        return global_pred
