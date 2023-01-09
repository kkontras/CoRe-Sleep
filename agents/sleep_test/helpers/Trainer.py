import torch
import time

import numpy as np
from tqdm import tqdm
import einops
#
from agents.sleep_test.helpers.Shuffler import Shuffler
from agents.sleep_test.helpers.Consecutives_Predictor import Consecutives_Predictor
from collections import defaultdict
from colorama import Fore, Back, Style
from utils.config import process_config
from graphs.models.attention_models.windowFeature_base import *
from sklearn.metrics import f1_score, cohen_kappa_score
import optuna


class Trainer():

    def __init__(self, agent):
        self.agent = agent
        self.shuffler = Shuffler(self.agent.config.random_shuffling) if "random_shuffling" in self.agent.config else Shuffler()
        self.consecutives_predictor = Consecutives_Predictor(agent=agent)
        self.train_step_func = self._find_train_step_func()
        self.this_train_step_func = getattr(self, self.train_step_func)
        self._get_loss_weights()

    def sleep_train_step(self, trial=None):

        self.agent.model.train()
        self._freeze_encoders(config_model=self.agent.config.model, model=self.agent.model)
        self.agent.mem_loader._my_numel(self.agent.model, only_trainable=True)
        self.agent.start = time.time()

        tts, preds, incomplete_idxs, batch_loss, early_stop = [], [], [], [], False
        val_loss = {"combined":0}
        saved_at_step, prev_epoch_time = 0, 0
        for self.agent.logs["current_epoch"] in range(self.agent.logs["current_epoch"], self.agent.config.early_stopping.max_epoch):
            pbar = tqdm(enumerate(self.agent.data_loader.train_loader), desc="Training", leave=None, disable=self.agent.config.training_params.tdqm_disable, position=0)
            for batch_idx, served_dict in pbar:

                served_dict["data"] = {view: served_dict["data"][view].float().to(self.agent.device) for view in served_dict["data"]}
                served_dict["label"] = served_dict["label"][list(served_dict["label"].keys())[0]].flatten(start_dim=0, end_dim=1).to(self.agent.device)

                served_dict, perms = self.shuffler(served_dict)

                if len(served_dict["label"] .shape) > 1 and "softlabels" not in self.agent.config.dataset and not self.agent.config.dataset.softlabels:
                    served_dict["label"] = served_dict["label"] .argmax(dim=1)

                self.agent.optimizer.zero_grad()

                loss, pred, label, incomplete_idx  = self.this_train_step_func(served_dict)
                # torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), 1)
                self.agent.optimizer.step()
                self.agent.scheduler.step(self.agent.logs["current_step"]+1)
                # if "sparse_loss" in self.config and self.config.sparse_loss:
                # loss, pred = self.sleep_train_one_step_sparse(data, target, inits)

                batch_loss.append(loss)
                tts.append(label)
                preds.append(pred)
                incomplete_idxs.append(incomplete_idx)

                del served_dict, pred, loss, label, incomplete_idx
                torch.cuda.empty_cache()

                pbar_message = Fore.WHITE + "Training batch {0:d}/{1:d} steps no improve {2:d} with ".format(batch_idx, len(self.agent.data_loader.train_loader)-1, self.agent.logs["steps_no_improve"])
                mean_batch = self._calc_mean_batch_loss(batch_loss=batch_loss)
                for mean_key in mean_batch: pbar_message += "{}: {:.3f} ".format(mean_key, mean_batch[mean_key])

                if self.agent.logs["current_step"] % self.agent.config.early_stopping.validate_every == 0 and \
                    self.agent.logs["current_step"] // self.agent.config.early_stopping.validate_every >= self.agent.config.early_stopping.validate_after and \
                    batch_idx!=0:
                    early_stop, val_loss = self.agent.monitor_n_saver.checkpointing(batch_loss = mean_batch, predictions = preds, targets = tts, incomplete_idxs=incomplete_idxs)
                    batch_loss, tts, preds, incomplete_idxs = [], [], [], []
                    #OPTUNA Hyperparameter search
                    if trial:
                        trial.report(val_loss["total"], self.agent.logs["current_step"] // self.agent.config.early_stopping.validate_every)
                        # Handle pruning based on the intermediate value.
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                    if early_stop: return
                    saved_at_step = self.agent.logs["saved_step"] // self.agent.config.early_stopping.validate_every
                    prev_epoch_time = time.time() - self.agent.start
                    self.agent.start = time.time()


                for mean_key in val_loss: pbar_message += " val {}: {:.3f} ".format(mean_key, val_loss[mean_key])

                pbar_message += " time {:.1f} sec/step, ".format(prev_epoch_time)
                pbar_message += "saved at {}".format(saved_at_step)
                pbar.set_description(pbar_message)
                pbar.refresh()
                self.agent.logs["current_step"] += 1

    def sleep_train_one_epoch(self):
            """
            One epoch of training
            :return:
            """
            self.agent.model.train()
            for enc in range(len(self.agent.config.encoder_models)):
                if self.agent.config.freeze_encoders[i]:
                    if hasattr(self.agent.model.module,"enc_{}".format(i)):
                        for p in getattr(self.agent.model.module,"enc_{}".format(enc)).parameters():
                            p.requires_grad = False

            batch_loss = 0
            tts, preds = [], []
            pbar = tqdm(enumerate(self.agent.data_loader.train_loader), desc = "Training", leave=False, disable=self.agent.config.training_params.tdqm_disable)
            for batch_idx, (data, target, _, idxs) in pbar: #tqdm(enumerate(self.data_loader.train_loader), "Training", leave=False, disable=self.config.tdqm_disable):
                views = [data[i].float().to(self.agent.device) for i in range(len(data))]
                target = target.to(self.agent.device).flatten()
                self.agent.optimizer.zero_grad()
                pred = self.agent.model(views)
                loss = self.agent.loss(pred, target)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                loss.backward()
                #update progress bar
                pbar.set_description("Training batch {0:d}/{1:d} with loss {2:.5f}".format(batch_idx,len(self.agent.data_loader.train_loader),loss.item()))
                pbar.refresh()

                batch_loss += loss
                tts.append(target)
                preds.append(pred)
                self.agent.optimizer.step()
                self.agent.scheduler.step()
            # self.model.module.enc_0.dy_conv_0.update_temperature()
            # self.model.module.enc_0.dy_conv_1.update_temperature()
            # self.model.module.enc_0.dy_conv_3.update_temperature()
            tts = torch.cat(tts).cpu().numpy()
            preds = torch.cat(preds).argmax(axis=1).cpu().numpy()
            return batch_loss / len(self.agent.data_loader.train_loader), np.equal(tts, preds).sum() / len(tts), f1_score(preds, tts, average="macro"), cohen_kappa_score(preds, tts, average="macro")
    def sleep_train_one_step(self, served_dict):

            data = served_dict["data"]
            teacher_preds = served_dict["teacher_preds"] if "teacher_preds" in served_dict else None
            target = served_dict["label"]
            inits = served_dict["init"]

            # output = self.get_predictions_time_series(data, inits)
            output = self.agent.model(data)

            if teacher_preds:
                teacher_preds = teacher_preds.to(self.agent.device).flatten(start_dim=0, end_dim=1)
                loss = self.agent.loss(output["preds"]["combined"], target, teacher_preds)
            else:
                loss = self.agent.loss(output["preds"]["combined"], target)

            loss.backward()

            return {"total":loss.detach().cpu().numpy()}, {"combined": output["preds"]["combined"].detach().cpu().numpy()}, target, None
    # def sleep_train_one_step_alignment_order(self, data, target, inits, teacher_preds=None):
    #
    #         if "multi_loss_weights" in self.agent.config.model.args.multi_loss:
    #             w_supervised_loss = self.agent.config.model.args.multi_loss.multi_loss_weights["supervised_loss"] if "supervised_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
    #             w_alignments_loss = self.agent.config.model.args.multi_loss.multi_loss_weights["alignment_loss"] if "alignment_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
    #             w_order_loss = self.agent.config.model.args.multi_loss.multi_loss_weights["order_loss"] if "order_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
    #         else:
    #             w_supervised_loss, w_alignments_loss, w_order_loss = 1, 1, 1
    #
    #         return_matches= True if w_alignments_loss!=0 else False
    #         return_order= True if w_order_loss!=0 else False
    #
    #         # pred, matches = self.get_predictions_time_series_alignment(views, inits)
    #         output = self.agent.model(data, return_matches=return_matches, return_order=return_order)
    #
    #         if "kd_label" in self.agent.config.dataset and self.agent.config.dataset.kd_label:
    #             teacher_preds = teacher_preds.to(self.agent.device).flatten(start_dim=0, end_dim=1)
    #             ce_loss = self.agent.loss(output[0], target, teacher_preds)
    #         else:
    #             ce_loss = self.agent.loss(output[0], target)
    #
    #         total_loss = w_supervised_loss * ce_loss
    #
    #         if w_alignments_loss!=0:
    #
    #             matches = output[1].flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
    #             if "blip_loss" in self.agent.config.model:
    #                 alignment_target = self.agent.alignment_target[:data[0].shape[0], :data[0].shape[1], :data[0].shape[1]].flatten(start_dim=0, end_dim=1)
    #             else:
    #                 alignment_target = self.agent.alignment_target[:data[0].shape[0], :data[0].shape[1]].flatten(start_dim=0, end_dim=1)
    #
    #             alignment_loss = self.agent.alignment_loss(matches, alignment_target)
    #             total_loss += w_alignments_loss*alignment_loss
    #         else:
    #             alignment_loss = torch.tensor(0).cuda()
    #
    #         if w_order_loss!=0:
    #             unfolded_target = einops.rearrange(target," (b outer) -> b outer", b=data[0].shape[0], outer=data[0].shape[1])
    #             unfolded_target = unfolded_target.unfold(1,3,1)
    #             same_label_left = unfolded_target[:, :, 0] == unfolded_target[:, :, 1]
    #             same_label_right = unfolded_target[:, :, 2] == unfolded_target[:, :, 1]
    #
    #             order_target = torch.zeros([data[0].shape[0], data[0].shape[1]-2]).cuda() != 0 #Initially everything is False
    #             order_target[same_label_left] = same_label_right[same_label_left] == True #If the ones that are left are True, index right and take only the True ones from that.
    #
    #             order_target = order_target.flatten().long()
    #             index_output=1
    #             if return_matches: index_output+=1
    #             order_loss = self.agent.order_loss(output[index_output], order_target)
    #             total_loss += w_order_loss*order_loss
    #         else:
    #             order_loss = torch.tensor(0).cuda()
    #
    #         total_loss.backward()
    #
    #         return [total_loss, ce_loss, alignment_loss, order_loss], output[0], target
    def sleep_train_one_step_alignment_order_multisupervised(self, served_dict):

            return_matches= True if self.w_loss["alignments"]!=0 else False
            return_order= True if self.w_loss["order"]!=0 else False
            return_consistency= True if self.w_loss["consistency"]!=0 else False
            return_reconstruction= True if self.w_loss["reconstruction"]!=0 else False

            clean_train = self.agent.config.model.args.clean_train if "clean_train" in self.agent.config.model.args else False

            # pred, matches = self.get_predictions_time_series_alignment(views, inits)
            data = served_dict["data"]
            teacher_preds = served_dict["teacher_preds"] if "teacher_preds" in served_dict else None
            target = served_dict["label"]
            inits = served_dict["init"]

            if "augment_within" in self.agent.config.model.args and self.agent.config.model.args.augment_within:
                output = self.agent.model(data, return_matches=return_matches, return_order=return_order, return_reconstruction=return_reconstruction, skip_modality="full")
                output["preds"]["eeg_2"] = self.agent.model(data, return_matches=return_matches, return_order=return_order, return_reconstruction=return_reconstruction,  skip_modality="eog")["preds"]["eeg"]
                output["preds"]["eog_2"] = self.agent.model(data, return_matches=return_matches, return_order=return_order, return_reconstruction=return_reconstruction, skip_modality="eeg")["preds"]["eog"]

            elif "augment_repeat" in self.agent.config.model.args and self.agent.config.model.args.augment_repeat:
                output = self.agent.model(data, return_matches=return_matches, return_order=return_order, return_reconstruction=return_reconstruction, skip_modality="full")
                output_2 = self.agent.model(data, return_matches=return_matches, return_order=return_order, return_reconstruction=return_reconstruction, skip_modality="full")
                output_3 = self.agent.model(data, return_matches=return_matches, return_order=return_order, return_reconstruction=return_reconstruction, skip_modality="full")
                for i in output["preds"]:
                    output["preds"][i] = (output["preds"][i] + output_2["preds"][i] + output_3["preds"][i])/3
            else:
                output = self.get_predictions_time_series_onlyskip(views=data, inits=inits, skip_modality=served_dict["skip_view"], return_matches=return_matches)
                # output = self.get_predictions_time_series(views=data, inits=inits, skip_modality=served_dict["skip_view"], return_matches=return_matches)
                # output = self.agent.model(data, return_matches=return_matches, return_order=return_order, return_reconstruction=return_reconstruction)
            teacher_preds = teacher_preds.to(self.agent.device).flatten(start_dim=0, end_dim=1)  if "kd_label" in self.agent.config and self.agent.config.kd_label else None

            total_loss =  torch.zeros(1).squeeze().to(self.agent.device)
            output_losses, ce_loss = {}, {}
            if "preds" not in output: output["preds"] = {}

            for k, v in output["preds"].items():
                if self.w_loss[k]!=0:
                    this_target = target
                    if "incomplete_idx" in output:
                        this_target = this_target[output["incomplete_idx"][k].flatten().bool()]

                    if len(this_target)>0: #TODO: Check if this one needs to be one or zero
                        ce_loss[k] = self.agent.loss(v, this_target, teacher_preds) if teacher_preds else self.agent.loss(v, this_target)
                        total_loss += self.w_loss[k] * ce_loss[k]
                        ce_loss[k] = ce_loss[k].detach().cpu().numpy()
                        output_losses.update({"ce_loss_{}".format(k): ce_loss[k]})

            consistency_loss = 0
            if return_consistency:
                for pred_i in range(len(output["preds"].keys())):
                    for pred_j in range(pred_i+1, len(output["preds"].keys())):
                        pred_i_key = list(output["preds"].keys())[pred_i]
                        pred_j_key = list(output["preds"].keys())[pred_j]
                        if self.w_loss[pred_i_key] != 0 or self.w_loss[pred_j_key]!=0:
                            consistency_loss += (self.w_loss[pred_i_key]*self.w_loss[pred_j_key])*self.agent.consistency_loss(output["preds"][pred_i_key], output["preds"][pred_j_key])
                            # consistency_loss += (self.w_loss[pred_i_key]*self.w_loss[pred_j_key])*self.agent.consistency_loss(output["preds"][pred_j_key], output["preds"][pred_i_key])

                total_loss += self.w_loss["consistency"]*consistency_loss
                output_losses["consistency_loss"] = consistency_loss.detach().cpu().numpy()
            if return_matches:
                matches = output["matches"]
                if matches is not None and type(matches) is dict and "stft_eeg" in matches and matches["stft_eeg"] is not None:
                    if len(matches["stft_eeg"].shape)==2:
                        alignment_target = torch.arange(matches["stft_eeg"].shape[0]).to(matches["stft_eeg"].device)
                    else:
                        alignment_target = torch.arange(matches["stft_eeg"].shape[1]).tile(matches["stft_eeg"].shape[0]).to(matches["stft_eeg"].device)
                        matches["stft_eeg"] = matches["stft_eeg"].flatten(start_dim=0, end_dim=1)
                        matches["stft_eog"] = matches["stft_eog"].flatten(start_dim=0, end_dim=1)

                    # if "target_mask" in output:
                    #     if (output["target_mask"]["combined"]==0).sum()>0 :
                    #         alignment_loss = self.agent.alignment_loss(matches["stft_eeg"][output["target_mask"]["combined"]==0], alignment_target[output["target_mask"]["combined"]==0])
                    #         alignment_loss += self.agent.alignment_loss(matches["stft_eog"][output["target_mask"]["combined"]==0], alignment_target[output["target_mask"]["combined"]==0])
                    #         total_loss += self.w_loss["alignments"] * alignment_loss
                    #         alignment_loss = alignment_loss.detach().cpu().numpy()
                    #         output_losses.update({"alignment_loss": alignment_loss})
                    #         del alignment_loss
                    #     else:
                    #         pass
                    # else :
                    alignment_loss = self.agent.alignment_loss(matches["stft_eeg"], alignment_target)
                    alignment_loss += self.agent.alignment_loss(matches["stft_eog"], alignment_target)
                    total_loss += self.w_loss["alignments"]*alignment_loss
                    alignment_loss = alignment_loss.detach().cpu().numpy()
                    output_losses.update({"alignment_loss": alignment_loss})
                    del alignment_loss
                else:
                    output_losses.update({"alignment_loss": np.array(0, dtype=np.float32)})
            if return_order:
                unfolded_target = einops.rearrange(target," (b outer) -> b outer", b=data[0].shape[0], outer=data[0].shape[1])
                unfolded_target = unfolded_target.unfold(1,3,1)
                same_label_left = unfolded_target[:, :, 0] == unfolded_target[:, :, 1]
                same_label_right = unfolded_target[:, :, 2] == unfolded_target[:, :, 1]

                order_target = torch.zeros([data[0].shape[0], data[0].shape[1]-2]).cuda() != 0 #Initially everything is False
                order_target[same_label_left] = same_label_right[same_label_left] == True #If the ones that are left are True, index right and take only the True ones from that.

                order_target = order_target.flatten().long()
                order_loss = self.agent.order_loss(output["order"], order_target)
                total_loss += self.w_loss["order"]*order_loss
                order_loss = order_loss.detach().cpu().numpy()
                output_losses.update({"order_loss": order_loss})
            if return_reconstruction:

                reconstruction_loss_eeg = self.agent.reconstruction_loss(reconstruction=output["reconstruction"]["eeg"], input=output["input"]["eeg"])
                reconstruction_loss_eog = self.agent.reconstruction_loss(reconstruction=output["reconstruction"]["eog"], input=output["input"]["eog"])
                total_loss += self.w_loss["reconstruction"]*(reconstruction_loss_eeg["total"]+reconstruction_loss_eog["total"])
                output_losses.update({"recon_loss_eeg": reconstruction_loss_eeg["total"].detach().cpu().numpy(), "recon_loss_eog": reconstruction_loss_eog["total"].detach().cpu().numpy()})

            if total_loss.requires_grad:
                total_loss.backward()
            else:
                raise ValueError("Here we have a problem, the total loss is empty or does not require gradient, output_losses: {}".format(output_losses))

            total_loss =  total_loss.detach().cpu().numpy()
            output_losses.update({"total": total_loss})
            del total_loss

            for i in output["preds"]:  output["preds"][i] =  output["preds"][i].detach().cpu().numpy()

            if "incomplete_idx" not in output: output["incomplete_idx"] = None

            return output_losses, output["preds"], target, output["incomplete_idx"]

    def sleep_train_one_step_adv(self, data, target, inits, teacher_preds=None):

            torch.autograd.set_detect_anomaly(True)
            for i in data: i.requires_grad = True

            # pred = self.get_predictions_time_series(data, inits)
            output = self.agent.model(data)

            if teacher_preds:
                teacher_preds = teacher_preds.to(self.agent.device).flatten(start_dim=0, end_dim=1)
                loss = self.agent.loss(output["preds"]["combined"], target, teacher_preds)
            else:
                loss = self.agent.loss(output["preds"]["combined"], target)

            loss.backward(retain_graph=True)
            data_plus = [view + self.agent.config.training_params.adversarial_training.adv_epsilon * (view.grad).sign() for view in data]
            self.agent.optimizer.zero_grad()
            output_plus = self.agent.model(data_plus)
            if teacher_preds:
                teacher_preds = teacher_preds.to(self.agent.device).flatten(start_dim=0, end_dim=1)
                loss_plus = self.agent.loss(output_plus["preds"]["combined"], target, teacher_preds)
            else:
                loss_plus = self.agent.loss(output_plus["preds"]["combined"], target)
            loss_total = loss_plus + loss
            loss_total.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            return {"total":loss.detach().cpu().numpy(), "total_adv": loss_plus.detach().cpu().numpy()}, {"combined": output["preds"]["combined"].detach().cpu().numpy()}, target

    def sleep_train_one_step_alignment_order_adv(self, data, target, inits, teacher_preds=None):

            torch.autograd.set_detect_anomaly(True)

            for i in data: i.requires_grad = True
            # pred = self.model(views)
            # pred = self.get_predictions_time_series(views, inits)
            pred, matches, order_pred = self.agent.model(data, return_matches=True, return_order=True)

            if teacher_preds:
                teacher_preds = teacher_preds.to(self.agent.device).flatten(start_dim=0, end_dim=1)
                ce_loss = self.agent.loss(pred, target, teacher_preds)
            else:
                ce_loss = self.agent.loss(pred, target)

            matches = matches.flatten(start_dim=0, end_dim=1)

            if "blip_loss" in self.agent.config:
                alignment_target = self.agent.alignment_target[:data[0].shape[0], :data[0].shape[1], :data[0].shape[1]].flatten(start_dim=0, end_dim=1)

            else:
                alignment_target = self.agent.alignment_target[:data[0].shape[0], :data[0].shape[1]].flatten(start_dim=0, end_dim=1)
            alignment_loss = self.agent.alignment_loss(matches, alignment_target)

            unfolded_target = einops.rearrange(target," (b outer) -> b outer", b=32, outer=21)
            unfolded_target = unfolded_target.unfold(1,3,1)
            same_label_left = unfolded_target[:, :, 0] == unfolded_target[:, :, 1]
            same_label_right = unfolded_target[:, :, 2] == unfolded_target[:, :, 1]

            order_target = torch.zeros([32, 19]).cuda() != 0 #Initially everything is False
            order_target[same_label_left] = same_label_right[same_label_left] == True #If the ones that are left are True, index right and take only the True ones from that.

            order_target = order_target.flatten().long()
            order_loss = self.agent.order_loss(order_pred, order_target)

            if "multi_loss_weights" in self.agent.config.model.args.multi_loss:
                w_supervised_loss = self.agent.config.model.args.multi_loss.multi_loss_weights["supervised_loss"] if "supervised_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
                w_alignments_loss = self.agent.config.model.args.multi_loss.multi_loss_weights["alignment_loss"] if "alignment_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
                w_order_loss = self.agent.config.model.args.multi_loss.multi_loss_weights["order_loss"] if "order_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
            else:
                w_supervised_loss, w_alignments_loss, w_order_loss = 1, 1, 1

            total_loss = w_supervised_loss*ce_loss + w_alignments_loss*alignment_loss + w_order_loss*order_loss

            total_loss.backward(retain_graph=True)

            data_plus = [view + self.agent.config.adv_epsilon * (view.grad).sign() for view in data]

            self.agent.optimizer.zero_grad()
            # pred_plus = self.get_predictions_time_series(views_plus, inits)
            pred_plus, matches_plus, order_pred_plus = self.agent.model(data_plus, return_matches=True, return_order=True)

            ce_loss_plus = self.agent.loss(pred_plus, target)
            alignment_loss_plus = self.agent.alignment_loss(matches_plus, alignment_target)
            order_loss_plus = self.agent.alignment_loss(order_pred, order_target)

            total_loss_plus = w_supervised_loss*ce_loss_plus + w_alignments_loss*alignment_loss_plus + w_order_loss*order_loss_plus
            total_loss_plus.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)


            return [total_loss_plus, total_loss, ce_loss_plus, ce_loss, alignment_loss_plus, alignment_loss, order_loss_plus, order_loss], pred
    def sleep_train_one_step_sparse(self, data, target, inits, current_epoch=0):
            raise ValueError("This method has not been checked whether it works, please remember what was the point and implrement it well.")

            views = [data[i].float().to(self.device) for i in range(len(data))]
            target = target.to(self.device).flatten()
            forward_hook_manager = ForwardHookManager(self.device)
            forward_hook_manager.add_hook(self.model,
                                          'inner_tf_mod0_l3_RA.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention',
                                          requires_input=False, requires_output=True)
            forward_hook_manager.add_hook(self.model,
                                          'outer_tf_mod0_l3_RA.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention',
                                          requires_input=False, requires_output=True)
            forward_hook_manager.add_hook(self.model,
                                          'inner_tf_mod0_l0_RA.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention',
                                          requires_input=False, requires_output=True)
            forward_hook_manager.add_hook(self.model,
                                          'outer_tf_mod0_l0_RA.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention',
                                          requires_input=False, requires_output=True)
            forward_hook_manager.add_hook(self.model,
                                          'inner_tf_mod0_l1_RA.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention',
                                          requires_input=False, requires_output=True)
            forward_hook_manager.add_hook(self.model,
                                          'outer_tf_mod0_l1_RA.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention',
                                          requires_input=False, requires_output=True)
            forward_hook_manager.add_hook(self.model,
                                          'inner_tf_mod0_l2_RA.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention',
                                          requires_input=False, requires_output=True)
            forward_hook_manager.add_hook(self.model,
                                          'outer_tf_mod0_l2_RA.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention',
                                          requires_input=False, requires_output=True)

            self.optimizer.zero_grad()
            pred = self.get_predictions_time_series(views, inits)
            loss = self.loss(pred, target)
            io_dict = forward_hook_manager.pop_io_dict()
            inner_weights = torch.cat([
                io_dict['inner_tf_mod0_l0_RA.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1],
                io_dict['inner_tf_mod0_l1_RA.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1],
                io_dict['inner_tf_mod0_l2_RA.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1]], dim=2)

            cls_att_w = io_dict['inner_tf_mod0_l3_RA.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1][0]

            outer_weights = torch.cat([
                io_dict['outer_tf_mod0_l0_RA.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1],
                io_dict['outer_tf_mod0_l1_RA.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1],
                io_dict['outer_tf_mod0_l2_RA.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1],
                io_dict['outer_tf_mod0_l3_RA.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1]], dim=2)

            loss -= 0.00001 * (torch.norm(inner_weights)+torch.norm(outer_weights)+torch.norm(cls_att_w))

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step(current_epoch)

            return loss, pred

    def _get_loss_weights(self):

        if ("multi_loss" in self.agent.config.model.args and "renew_each_step" in self.agent.config.model.args.multi_loss and self.agent.config.model.args.multi_loss.renew_each_step) or not hasattr(self.agent.logs,"w_loss"):
            w_loss = defaultdict(int)
            if "multi_loss" in self.agent.config.model.args and "multi_loss_weights" in self.agent.config.model.args.multi_loss:

                if "multi_supervised_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights:
                    for k, v in self.agent.config.model.args.multi_loss.multi_loss_weights.multi_supervised_loss.items():
                        w_loss[k] = v
                w_loss["alignments"] = self.agent.config.model.args.multi_loss.multi_loss_weights["alignment_loss"] if "alignment_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
                w_loss["order"] = self.agent.config.model.args.multi_loss.multi_loss_weights["order_loss"] if "order_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
                w_loss["consistency"] = self.agent.config.model.args.multi_loss.multi_loss_weights["consistency_loss"] if "consistency_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
                w_loss["reconstruction"] = self.agent.config.model.args.multi_loss.multi_loss_weights["reconstruction"] if "reconstruction" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
            else:
                w_loss["total"]= 1
                # raise Warning("We dont have multi supervised loss weights")
            self.w_loss = w_loss
            self.agent.logs["w_loss"] = w_loss
        elif hasattr(self.agent.logs,"w_loss") and self.agent.config.model.load_ongoing:
            self.w_loss = self.agent.logs.w_loss
        print("Loss Weights are", dict(self.w_loss))
    def _calc_mean_batch_loss(self, batch_loss):
        mean_batch = defaultdict(list)
        for b_i in batch_loss:
            for loss_key in b_i:
                mean_batch[loss_key].append(b_i[loss_key])
        for key in mean_batch:
            mean_batch[key] = np.array(mean_batch[key]).mean(axis=0)
        return mean_batch
    def _freeze_encoders(self, config_model, model):
        for enc in range(len(config_model.encoders)):
            if "freeze_encoder" in config_model.encoders[enc] and config_model.encoders[enc]["freeze_encoder"]:
                if hasattr(model.module, "enc_{}".format(enc)):
                    print("Freezing encoder enc_{}".format(enc))
                    for p in getattr(model.module, "enc_{}".format(enc)).parameters():
                        p.requires_grad = False
            if "encoders" in config_model.encoders[enc]:
                for enc_i in range(len(config_model.encoders)):
                    self._freeze_encoders(config_model = config_model.encoders[enc_i], model = getattr(model.module, "enc_{}".format(enc_i)))

    def _find_train_step_func(self):

        if "training_type" not in self.agent.config.model.args or self.agent.config.model.args.training_type == "normal":
            train_step_func = "sleep_train_one_step_adv" if "adversarial_training" in self.agent.config.training_params and self.agent.config.training_params.adversarial_training.use else "sleep_train_one_step"
        elif self.agent.config.model.args.training_type == "alignment":
            train_step_func = "sleep_train_one_step_alignment_adv" if "adversarial_training" in self.agent.config.training_params and self.agent.config.training_params.adversarial_training.use else "sleep_train_one_step_alignment"
        elif self.agent.config.model.args.training_type == "alignment_order":
            train_step_func = "sleep_train_one_step_alignment_order_adv" if "adversarial_training" in self.agent.config.training_params and self.agent.config.training_params.adversarial_training.use else "sleep_train_one_step_alignment_order"
        elif self.agent.config.model.args.training_type == "alignment_order_multisupervised":
            train_step_func = "sleep_train_one_step_alignment_order_multisupervised_adv" if "adversarial_training" in self.agent.config.training_params and self.agent.config.training_params.adversarial_training.use else "sleep_train_one_step_alignment_order_multisupervised"
        elif self.agent.config.model.args.training_type == "router":
            train_step_func = "sleep_train_one_step_router"
        elif self.agent.config.model.args.training_type == "reconstruction":
            train_step_func = "sleep_train_one_step_reconstruction"
        else:
            raise ValueError("Training type does not exist, check self.agent.config.model.training_type! Available ones are 'normal', 'alignment' and 'alignment_order' ")

        print("Training function is {}".format(train_step_func))
        return train_step_func


    def get_predictions_time_series_onlyskip(self, views, inits, skip_modality=None, **kwargs):

        """
        This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
        :param views: List of tensors, data views/modalities
        :param inits: Tensor indicating with value one, when there incontinuities.
        :return: predictions of the self.model on the batch
        """

        this_view = views[list(inits.keys())[0]]

        this_skip_modality = torch.zeros(this_view.shape[0], this_view.shape[1])
        if type(skip_modality)==dict and "stft_eeg" and skip_modality["stft_eeg"].shape[0]*skip_modality["stft_eeg"].shape[1]>0 and "stft_eog" and skip_modality["stft_eog"].shape[0]*skip_modality["stft_eog"].shape[1]>0:
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

    def get_predictions_time_series(self, views, inits, skip_modality=None, **kwargs):




        """
        This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
        :param views: List of tensors, data views/modalities
        :param inits: Tensor indicating with value one, when there incontinuities.
        :return: predictions of the self.model on the batch
        """
        this_inits = inits[list(inits.keys())[0]]
        this_view = views[list(inits.keys())[0]]

        if skip_modality:
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
            pred ={"preds":{"combined": torch.zeros(batch * outer, 5).to(this_view.device),
                            "eeg": torch.zeros(batch * outer, 5).to(this_view.device),
                            "eog": torch.zeros(batch * outer, 5).to(this_view.device)},
                   "matches": {"stft_eeg": torch.zeros(batch, outer, outer).to(this_view.device),
                               "stft_eog": torch.zeros(batch, outer, outer).to(this_view.device)}
                   }
            for batch_idx in inits_sum_batch:
                ones_idx = (this_inits[batch_idx] > 0).nonzero(as_tuple=True)[0]
                if (ones_idx[0] + 1 == ones_idx[1]  ): #and ones_idx[0]!=0 and ones_idx[1]!= len(inits[idx])
                    if ones_idx[0] == 0:
                        this_skip_modality = {view: skip_modality[view][batch_idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views} if skip_modality else None
                        pred_split_0 = self.agent.model({view: views[view][batch_idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views}, skip_modality=this_skip_modality, **kwargs)
                    else:
                        this_skip_modality = {view: skip_modality[view][batch_idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views} if skip_modality else None
                        pred_split_0 = self.agent.model({view: views[view][batch_idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views}, skip_modality=this_skip_modality, **kwargs)

                    if ones_idx[1] == len(this_inits[batch_idx]):
                        this_skip_modality = {view: skip_modality[view][batch_idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views} if skip_modality else None
                        pred_split_1 = self.agent.model({view: views[view][batch_idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views}, skip_modality=this_skip_modality, **kwargs)
                    else:
                        this_skip_modality = {view: skip_modality[view][batch_idx, ones_idx[1]:].unsqueeze(dim=0) for view in views} if skip_modality else None
                        pred_split_1 = self.agent.model({view: views[view][batch_idx, ones_idx[1]:].unsqueeze(dim=0) for view in views}, skip_modality=this_skip_modality, **kwargs)

                    if "combined" in pred_split_0["preds"] and "combined" in pred_split_1["preds"]:
                        pred["preds"]["combined"][batch_idx * outer:(batch_idx + 1) * outer] = torch.cat([pred_split_0["preds"]["combined"], pred_split_1["preds"]["combined"]], dim=0)
                    if "eog" in pred_split_0["preds"] and "eog" in pred_split_1["preds"]:
                        pred["preds"]["eog"][batch_idx * outer:(batch_idx + 1) * outer] = torch.cat(
                            [pred_split_0["preds"]["eog"], pred_split_1["preds"]["eog"]], dim=0)
                    if "eeg" in pred_split_0["preds"] and "eeg" in pred_split_1["preds"]:
                        pred["preds"]["eeg"][batch_idx * outer:(batch_idx + 1) * outer] = torch.cat(
                            [pred_split_0["preds"]["eeg"], pred_split_1["preds"]["eeg"]], dim=0)
                    if "matches" in pred:
                        if type(pred_split_0["matches"])==dict:
                            pred["matches"]["stft_eeg"][batch_idx, :ones_idx[1], :ones_idx[1]] = pred_split_0["matches"]["stft_eeg"]
                            pred["matches"]["stft_eog"][batch_idx, :ones_idx[1], :ones_idx[1]] = pred_split_0["matches"]["stft_eog"]
                        if type(pred_split_1["matches"]) == dict:
                            pred["matches"]["stft_eeg"][batch_idx, ones_idx[1]:, ones_idx[1]:] = pred_split_1["matches"]["stft_eeg"]
                            pred["matches"]["stft_eog"][batch_idx, ones_idx[1]:, ones_idx[1]:] = pred_split_1["matches"]["stft_eog"]

                else:
                    this_skip_modality = {view: skip_modality[view][batch_idx].unsqueeze(dim=0) for view in views} if skip_modality else None
                    current_preds = self.agent.model({view: views[view][batch_idx].unsqueeze(dim=0) for view in views}, skip_modality=this_skip_modality, **kwargs)
                    if "eeg" in current_preds["preds"]:
                        pred["preds"]["eeg"][batch_idx * outer:(batch_idx + 1) * outer] = current_preds["preds"]["eeg"]
                    if "eog" in current_preds["preds"]:
                        pred["preds"]["eog"][batch_idx * outer:(batch_idx + 1) * outer] = current_preds["preds"]["eog"]
                    if "combined" in current_preds["preds"]:
                        pred["preds"]["combined"][batch_idx * outer:(batch_idx + 1) * outer] = current_preds["preds"]["combined"]
                    if "matches" in pred:
                        if type(current_preds["matches"]) == dict:
                            pred["matches"]["stft_eeg"][batch_idx] = current_preds["matches"]["stft_eeg"]
                            pred["matches"]["stft_eog"][batch_idx] = current_preds["matches"]["stft_eog"]
                batch_idx_checked[batch_idx] = False

            this_skip_modality = {view: skip_modality[view][batch_idx_checked] for view in views} if skip_modality else None
            current_preds = self.agent.model({view: views[view][batch_idx_checked] for view in views}, skip_modality=this_skip_modality, **kwargs)
            if "eeg" in current_preds["preds"]:
                pred["preds"]["eeg"][batch_idx_checked.repeat_interleave(outer)] = current_preds["preds"]["eeg"]
            if "eog" in current_preds["preds"]:
                pred["preds"]["eog"][batch_idx_checked.repeat_interleave(outer)] = current_preds["preds"]["eog"]
            if "combined" in current_preds["preds"]:
                pred["preds"]["combined"][batch_idx_checked.repeat_interleave(outer)] = current_preds["preds"]["combined"]
            if "matches" in pred:
                if type(current_preds["matches"]) == dict:
                    pred["matches"]["stft_eeg"][batch_idx_checked] = current_preds["matches"]["stft_eeg"]
                    pred["matches"]["stft_eog"][batch_idx_checked] = current_preds["matches"]["stft_eog"]
        else:
            pred = self.agent.model(views, skip_modality=skip_modality, **kwargs)
        return pred
