import torch
import time

import numpy as np
from tqdm import tqdm
import einops
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix

from agents.sleep_test.helpers.Consecutives_Predictor import Consecutives_Predictor
from agents.sleep_test.helpers.Shuffler import Shuffler
from collections import defaultdict
from utils.config import process_config
from graphs.models.attention_models.windowFeature_base import *


# def sleep_load_encoder(encoders):
#     encs = []
#     for num_enc in range(len(encoders)):
#         if encoders[num_enc]["model"] == "TF":
#             layers = ["huy_pos_inner", "inner_att", "aggregation_att_contx_inner", "huy_pos_outer", "outer_att"]
#             enc = Multi_Transformer(128, inner= 29, outer = 21, modalities=1, heads=8,
#                                  layers = layers, num_layers=4, pos = False)
#         else:
#             enc_class = globals()[encoders[num_enc]["model"]]
#             args = encoders[num_enc]["args"τι ]
#             enc = enc_class(args = args)
#             enc = nn.DataParallel(enc, device_ids=[torch.device(0)])
#
#         if encoders[num_enc]["pretrainedEncoder"]["use"]:
#             print("Loading encoder from {}".format(encoders[num_enc]["pretrainedEncoder"]["dir"]))
#             checkpoint = torch.load(encoders[num_enc]["pretrainedEncoder"]["dir"])
#             enc.load_state_dict(checkpoint["encoder_state_dict"])
#         encs.append(enc)
#     return encs
#
# def load_models(config, device, checkpoint, only_model=False):
#
#     model_class = globals()[config.model.model_class]
#     # config.pretrainedEncoder = [False]
#     enc = sleep_load_encoder(encoders=config.model.encoders)
#     model = model_class(enc, args = config.model.args)
#     # model = model.to('cpu')
#     # model = nn.DataParallel(model, device_ids='cpu')
#     model = model.to(device)
#     model = nn.DataParallel(model, device_ids=[torch.device(i) for i in config.gpu_device])
#
#     #
#     if only_model:
#         return model
#
#     # config.pretrainedEncoder = [True]
#     # enc = sleep_load_encoder(encoder_models=config.encoder_models,pretrainedEncoder=config.pretrainedEncoder,save_dir_encoder=config.savetrainedEncoder)
#     # best_model = model_class(enc, channel = config.channel)
#     # best_model = best_model.to(device)
#     # best_model = nn.DataParallel(best_model, device_ids=[torch.device(i) for i in config.gpu_device])
#
#     best_model = copy.deepcopy(model)
#     # best_model = best_model.to('cpu')
#     # best_model = nn.DataParallel(best_model, device_ids='cpu')
#     # model.load_state_dict(checkpoint["model_state_dict"])
#     best_model.load_state_dict(checkpoint["best_model_state_dict"])
#
#     return model, best_model
#
# multimodal_config_name = "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw.json"
# eeg_config_name = "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_adv.json"
# eog_config_name = "./configs/shhs/single_channel/fourier_transformer_cls_eog_mat.json"
# emg_config_name = "./configs/shhs/single_channel/fourier_transformer_cls_emg_mat.json"
# router_config_name = "./configs/shhs/router/router_fourier_tf_eeg_eog.json"
#
# multimodal_config = process_config(multimodal_config_name, False)
# eeg_config = process_config(eeg_config_name, False)
# eog_config = process_config(eog_config_name, False)
# emg_config = process_config(emg_config_name, False)
# router_config = process_config(router_config_name, False)
# device = "cuda:0"
# #Load the models
# checkpoint_multimodal = torch.load(multimodal_config.model.save_dir, map_location="cpu")
# checkpoint_eeg = torch.load(eeg_config.model.save_dir, map_location="cpu")
# checkpoint_eog = torch.load(eog_config.model.save_dir, map_location="cpu")
# _, best_model_multimodal = load_models(config=multimodal_config, device=device, checkpoint=checkpoint_multimodal)
# _, best_model_eeg = load_models(config=eeg_config, device=device, checkpoint=checkpoint_eeg)
# _, best_model_eog = load_models(config=eog_config, device=device, checkpoint=checkpoint_eog)
#
# best_model_multimodal.eval()
# best_model_eeg.eval()
# best_model_eog.eval()

class Validator_Tester():
    def __init__(self, agent):
        self.agent = agent
        self.multi_supervised = False
        self.shuffler = Shuffler(self.agent.config.random_shuffling) if "random_shuffling" in self.agent.config else Shuffler()
        self.valtest_step_func = self._find_valtest_step_func()
        self.this_valtest_step_func = getattr(self, self.valtest_step_func)
        self._get_loss_weights()

    def sleep_validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.agent.model.eval()
        batch_loss = []
        tts, preds, inits = [], [], []
        # hidden = None
        with torch.no_grad():
            pbar = tqdm(enumerate(self.agent.data_loader.valid_loader), desc="Validation", leave=False,
                        disable=True, position=1)
            for batch_idx, served_dict in pbar:

                served_dict["data"] = {view: served_dict["data"][view].float().to(self.agent.device) for view in served_dict["data"]}
                served_dict["label"] = served_dict["label"][list(served_dict["label"].keys())[0]].flatten(start_dim=0, end_dim=1).to(self.agent.device)

                if len(served_dict["label"] .shape) > 1 and "softlabels" not in self.agent.config.dataset and not self.agent.config.dataset.softlabels:
                    served_dict["label"] = served_dict["label"] .argmax(dim=1)

                loss, pred, label = self.this_valtest_step_func(served_dict)

                batch_loss.append(loss)
                tts.append(label)
                preds.append(pred)
                # inits.append(init.flatten())

                del label, pred, loss, served_dict

                pbar_message = "Validation batch {0:d}/{1:d} with ".format(batch_idx,len(self.agent.data_loader.valid_loader) - 1)

                mean_batch = self._calc_mean_batch_loss(batch_loss = batch_loss)

                for mean_key in mean_batch: pbar_message += "{}: {:.3f} ".format(mean_key, mean_batch[mean_key])
                pbar.set_description(pbar_message)
                pbar.refresh()

            if "softlabels" in self.agent.config and self.agent.config.softlabels:
                tts = torch.cat(tts).argmax(dim=1).cpu().numpy()
            else:
                tts = torch.cat(tts).cpu().numpy()

            if self.agent.config.post_proc.val_postprocessing :
                preds = self.val_postprocessing(preds=preds, inits=inits)

            total_preds, val_metrics = {}, defaultdict(dict)
            val_metrics["val_loss"]= dict(mean_batch)
            for pred_key in preds[0]:
                total_preds[pred_key] = np.concatenate([pred[pred_key] for pred in preds], axis=0).argmax( axis=-1)
                val_metrics["val_acc"][pred_key] = np.equal(tts, total_preds[pred_key]).sum() / len(tts)
                val_metrics["val_f1"][pred_key] = f1_score(total_preds[pred_key], tts,average="macro")
                val_metrics["val_k"][pred_key] = cohen_kappa_score(total_preds[pred_key],tts)
                val_metrics["val_perclassf1"][pred_key] = f1_score(total_preds[pred_key], tts, average=None)
            val_metrics = dict(val_metrics)  # Avoid passing empty dicts to logs, better return an error!

        return val_metrics

    def sleep_test(self):
            """
            One cycle of model validation
            :return:
            """
            self.model.eval()
            batch_loss = []
            tts, preds, inits = [], [], []
            with torch.no_grad():
                pbar = tqdm(enumerate(self.data_loader.test_loader), desc="Test", leave=False,
                            disable=True, position=2)
                for batch_idx, batch in pbar:
                    data, target, init = batch[0], batch[1], batch[2]
                    data = [data[i].float().to(self.device) for i in range(len(data))]
                    if "softlabels" in self.config and self.config.dataset.softlabels:
                        target = target.to(self.device).flatten(start_dim=0,end_dim=1).float()
                    else:
                        target = target.to(self.device).flatten(start_dim=0,end_dim=1).long()

                    if "kd_label" in self.config and self.config.kd_label:
                        teacher_preds = batch[4]
                        teacher_preds = teacher_preds.to(self.device).flatten(start_dim=0, end_dim=1)

                    loss, pred, target = self.this_valtest_step_func(data, target, inits, teacher_preds)

                    tts.append(target)
                    preds.append(pred)
                    inits.append(init.flatten())
                    batch_loss.append(loss)

                    del data, target, pred, loss, init

                    pbar_message = "Test batch {0:d}/{1:d} with ".format(batch_idx,len(self.agent.data_loader.test_loader) - 1)

                    mean_batch = self._calc_mean_batch_loss(batch_loss = batch_loss)

                    for mean_key in mean_batch: pbar_message += "{}: {:.3f} ".format(mean_key, mean_batch[mean_key])
                    pbar.set_description(pbar_message)
                    pbar.refresh()

                if "softlabels" in self.agent.config.dataset and self.agent.config.dataset.softlabels:
                    tts = torch.cat(tts).argmax(dim=1).cpu().numpy()
                else:
                    tts = torch.cat(tts).cpu().numpy()
                preds = torch.cat(preds).cpu().numpy()
                if self.agent.config.post_proc.test_postprocessing :
                    inits = torch.cat(inits).numpy()
                    print("Test kappa without postprocessing is k= {0:.4f}".format(cohen_kappa_score(tts, preds.argmax(axis=1))))
                    w_idx = int(self.config.post_proc_step / 2 + 1)
                    while (w_idx < len(preds) - int(self.config.post_proc_step / 2 + 1)):

                        for n_class in range(len(preds[0])):
                            preds[w_idx, n_class] = preds[w_idx - int(self.config.post_proc_step / 2):w_idx + int(
                                self.config.post_proc_step / 2), n_class].sum() / self.config.post_proc_step
                        if (inits[int(w_idx + int(self.config.post_proc_step / 2 ))] == 1 ):
                            w_idx += int(self.config.post_proc_step) + 1
                        else:
                            w_idx+=1

                multiclass = False
                if preds.shape[1]>2:
                    multiclass = True
                preds = preds.argmax(axis=1)

                total_preds, test_metrics = {}, defaultdict(dict)
                test_metrics["test_loss"]= dict(mean_batch)
                for pred_key in preds[0]:
                    total_preds[pred_key] = np.concatenate([pred[pred_key] for pred in preds], axis=0).argmax( axis=-1)
                    test_metrics["test_acc"][pred_key] = np.equal(tts, total_preds[pred_key]).sum() / len(tts)
                    test_metrics["test_f1"][pred_key] = f1_score(total_preds[pred_key], tts,average="macro")
                    test_metrics["test_k"][pred_key] = cohen_kappa_score(total_preds[pred_key],tts)
                    test_metrics["test_f1_perclass"][pred_key] = f1_score(total_preds[pred_key], tts, average=None)
                test_metrics = dict(test_metrics)  # Avoid passing empty dicts to logs, better return an error!

            return test_metrics

    def val_postprocessing(self, inits, preds):

        print("This module needs update!")

        inits = torch.cat(inits).numpy()
        w_idx = int(self.agent.config.post_proc_step / 2 + 1)
        while (w_idx < len(preds) - int(self.agent.config.post_proc_step / 2 + 1)):

            for n_class in range(len(preds[0])):
                preds[w_idx, n_class] = preds[w_idx - int(self.agent.config.post_proc_step / 2):w_idx + int(
                    self.agent.config.post_proc_step / 2), n_class].sum() / self.agent.config.post_proc_step
            if (inits[int(w_idx + int(self.agent.config.post_proc_step / 2))] == 1):
                w_idx += int(self.agent.config.post_proc_step) + 1
            else:
                w_idx += 1
        return preds

    def _calc_mean_batch_loss(self, batch_loss):
        mean_batch = defaultdict(list)
        for b_i in batch_loss:
            for loss_key in b_i:
                mean_batch[loss_key].append(b_i[loss_key])
        for key in mean_batch:
            mean_batch[key] = np.array(mean_batch[key]).mean(axis=0)
        return mean_batch

    def _find_valtest_step_func(self):

        if "training_type" not in self.agent.config.model.args or self.agent.config.model.args.training_type == "normal":
            valtest_step_func = "sleep_valtest_one_step"
        elif self.agent.config.model.args.training_type == "alignment":
            valtest_step_func = "sleep_valtest_one_step_alignment_order"
        elif self.agent.config.model.args.training_type == "alignment_order":
            valtest_step_func = "sleep_valtest_one_step_alignment_order"
        elif self.agent.config.model.args.training_type == "alignment_order_multisupervised":
            valtest_step_func = "sleep_valtest_one_step_alignment_order_multisupervised"
        elif self.agent.config.model.args.training_type == "router":
            valtest_step_func = "sleep_valtest_one_step_router"
        elif self.agent.config.model.args.training_type == "reconstruction":
            valtest_step_func = "sleep_valtest_one_step_reconstruction"
        else:
            raise ValueError("Training type does not exist, check self.agent.config.model.training_type! Available ones are "
                             "'normal', 'router' and 'alignment_order' ")

        return valtest_step_func

    def sleep_valtest_one_step(self, served_dict):

            data = served_dict["data"]
            teacher_preds = served_dict["teacher_preds"] if "teacher_preds" in served_dict else None
            target = served_dict["label"]
            inits = served_dict["init"]

            output = self.get_predictions_time_series(data, inits)
            # output = self.agent.model(data)

            if teacher_preds:
                teacher_preds = teacher_preds.to(self.agent.device).flatten(start_dim=0, end_dim=1)
                loss = self.agent.loss(output["preds"]["combined"], target, teacher_preds)
            else:
                loss = self.agent.loss(output["preds"]["combined"], target)

            return {"total":loss.detach().cpu().numpy()}, {"combined": output["preds"]["combined"].detach().cpu().numpy()}, target

    def sleep_valtest_one_step_alignment_order(self, data, target, inits, teacher_preds=None):

            if "multi_loss_weights" in self.agent.config.model.args.multi_loss:
                w_supervised_loss = self.agent.config.model.args.multi_loss.multi_loss_weights["supervised_loss"] if "supervised_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
                w_alignments_loss = self.agent.config.model.args.multi_loss.multi_loss_weights["alignment_loss"] if "alignment_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
                w_order_loss = self.agent.config.model.args.multi_loss.multi_loss_weights["order_loss"] if "order_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0

            else:
                w_supervised_loss, w_alignments_loss, w_order_loss = 1, 1, 1

            return_matches= True if w_alignments_loss!=0 else False
            return_order= True if w_order_loss!=0 else False

            # pred, matches = self.get_predictions_time_series_alignment(views, inits)
            output = self.agent.model(data, return_matches=return_matches, return_order=return_order)

            if "kd_label" in self.agent.config.dataset and self.agent.config.dataset.kd_label:
                teacher_preds = teacher_preds.to(self.agent.device).flatten(start_dim=0, end_dim=1)
                ce_loss = self.agent.loss(output[0], target, teacher_preds)
            else:
                ce_loss = self.agent.loss(output[0], target)

            total_loss = (w_supervised_loss * ce_loss).cpu().numpy()


            if w_alignments_loss!=0:

                matches = output[1].flatten(start_dim=0, end_dim=1)
                if "blip_loss" in self.agent.config:
                    alignment_target = self.agent.alignment_target[:data[0].shape[0], :data[0].shape[1], :data[0].shape[1]].flatten(start_dim=0, end_dim=1)
                else:
                    alignment_target = self.agent.alignment_target[:data[0].shape[0], :data[0].shape[1]].flatten(start_dim=0, end_dim=1)

                alignment_loss = self.agent.alignment_loss(matches, alignment_target)
                total_loss += (w_alignments_loss*alignment_loss).cpu().numpy()
            else:
                alignment_loss = torch.tensor(0).cpu().numpy()

            if w_order_loss!=0:
                unfolded_target = einops.rearrange(target," (b outer) -> b outer", b=data[0].shape[0], outer=data[0].shape[1])
                unfolded_target = unfolded_target.unfold(1,3,1)
                same_label_left = unfolded_target[:, :, 0] == unfolded_target[:, :, 1]
                same_label_right = unfolded_target[:, :, 2] == unfolded_target[:, :, 1]

                order_target = torch.zeros([data[0].shape[0], data[0].shape[1]-2]).cuda() != 0 #Initially everything is False
                order_target[same_label_left] = same_label_right[same_label_left] == True #If the ones that are left are True, index right and take only the True ones from that.

                order_target = order_target.flatten().long()
                index_output=1
                if return_matches: index_output+=1
                order_loss = self.agent.order_loss(output[index_output], order_target)
                total_loss += (w_order_loss*order_loss).cpu().numpy()
            else:
                order_loss = torch.tensor(0).cpu().numpy()

            return {"total": total_loss, "ce_loss": ce_loss, "alignment_loss": alignment_loss, "order_loss": order_loss}, output[0], target

    def _get_loss_weights(self):

        if not hasattr(self,"w_loss") or ("multi_loss" in self.agent.config.model.args and "renew_each_step" in self.agent.config.model.args.multi_loss and self.agent.config.model.args.multi_loss.renew_each_step):
            w_loss = defaultdict(int)

            if "multi_loss" in self.agent.config.model.args and "multi_loss_weights" in self.agent.config.model.args.multi_loss:

                if "multi_supervised_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights:
                    for k, v in self.agent.config.model.args.multi_loss.multi_loss_weights.multi_supervised_loss.items():
                        w_loss[k] = v
                w_loss["alignments"] = self.agent.config.model.args.multi_loss.multi_loss_weights[
                    "alignment_loss"] if "alignment_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
                w_loss["order"] = self.agent.config.model.args.multi_loss.multi_loss_weights[
                    "order_loss"] if "order_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
                w_loss["consistency"] = self.agent.config.model.args.multi_loss.multi_loss_weights[
                    "consistency_loss"] if "consistency_loss" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0
                w_loss["reconstruction"] = self.agent.config.model.args.multi_loss.multi_loss_weights[
                    "reconstruction"] if "reconstruction" in self.agent.config.model.args.multi_loss.multi_loss_weights else 0

            else:
                w_loss["total"] = 1
                # raise Warning("We dont have multi supervised loss weights")
            self.w_loss = w_loss

    def sleep_valtest_one_step_alignment_order_multisupervised(self, served_dict):

            self._get_loss_weights()

            return_matches= True if self.w_loss["alignments"]!=0 else False
            return_order= True if self.w_loss["order"]!=0 else False
            return_consistency= True if self.w_loss["consistency"]!=0 else False
            return_reconstruction= True if self.w_loss["reconstruction"]!=0 else False


            data = served_dict["data"]
            target = served_dict["label"]
            inits = served_dict["init"]


            teacher_preds = served_dict["teacher_preds"] if "teacher_preds" in served_dict else None

            # pred, matches = self.get_predictions_time_series_alignment(views, inits)

            # if "three_modes" in self.agent.config.model.args and self.agent.config.model.args.three_modes:
            #     output = self.agent.model(data, return_matches=return_matches, return_order=return_order, return_reconstruction=return_reconstruction, skip_modality="full")
            #     output["preds"]["eeg"] = self.agent.model(data, return_matches=return_matches, return_order=return_order, return_reconstruction=return_reconstruction, skip_modality="eog")["preds"]["combined"]
            #     output["preds"]["eog"] = self.agent.model(data, return_matches=return_matches, return_order=return_order, return_reconstruction=return_reconstruction,skip_modality="eeg")["preds"]["combined"]
            #     if self.w_loss["emg"]:
            #         output["preds"]["emg"] = self.agent.model(data, return_matches=return_matches, return_order=return_order, return_reconstruction=return_reconstruction, skip_modality="emg")["preds"]["combined"]
            # else:
            output = self.get_predictions_time_series_onlyskip(views=data, inits=inits,
                                                               skip_modality=served_dict["skip_view"],
                                                               return_matches=return_matches)
            # output = self.agent.model(data, return_matches=return_matches, return_order=return_order, return_reconstruction=return_reconstruction)

            teacher_preds = teacher_preds.to(self.agent.device).flatten(start_dim=0, end_dim=1)  if "kd_label" in self.agent.config and self.agent.config.kd_label else None

            if "preds" not in output: output["preds"] = {}

            ce_loss = {}
            for k, v in output["preds"].items():
                ce_loss[k] = self.agent.loss(v, target, teacher_preds) if teacher_preds else self.agent.loss(v, target)

            total_loss = 0
            output_losses = {}
            for i in ce_loss:
                total_loss += self.w_loss[i] * ce_loss[i]
                ce_loss[i] = ce_loss[i].detach().cpu().numpy()
                output_losses.update({"ce_loss_{}".format(i): ce_loss[i]})

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


            if return_matches and not self.agent.config.model.args.ignore_alignment_val:
                matches = output["matches"]

                if matches is not None and type(matches)==dict:
                    if len(matches["stft_eeg"].shape)==2:
                        alignment_target = torch.arange(matches["stft_eeg"].shape[0]).to(matches["stft_eeg"].device)
                    elif len(matches["stft_eeg"].shape)==3:
                        alignment_target = torch.arange(matches["stft_eeg"].shape[1]).tile(matches["stft_eeg"].shape[0]).to(matches["stft_eeg"].device)
                        matches["stft_eeg"] = matches["stft_eeg"].flatten(start_dim=0, end_dim=1)
                        matches["stft_eog"] = matches["stft_eog"].flatten(start_dim=0, end_dim=1)
                    else:
                        print(matches["stft_eeg"].shape)
                    alignment_loss = self.agent.alignment_loss(matches["stft_eeg"], alignment_target)
                    alignment_loss += self.agent.alignment_loss(matches["stft_eog"], alignment_target)
                    total_loss += self.w_loss["alignments"]*alignment_loss
                    alignment_loss = alignment_loss.detach().cpu().numpy()
                    output_losses.update({"alignment_loss": alignment_loss})
                    del alignment_loss

                elif matches is not None and type(matches)== dict and len(matches.shape)==1:
                    output_losses.update({"alignment_loss": np.array(0, dtype=np.float32)})
                else:
                    alignment_loss = np.NaN
                    output_losses.update({"alignment_loss": alignment_loss})

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


            total_loss =  total_loss.detach().cpu().numpy()
            output_losses.update({"total": total_loss})

            for i in output["preds"]:  output["preds"][i] =  output["preds"][i].detach().cpu().numpy()

            return output_losses, output["preds"], target

    def sleep_valtest_one_step_router(self, data, target, inits, teacher_preds=None):

            return_matches= True if self.w_loss["alignments"]!=0 else False
            return_order= True if self.w_loss["order"]!=0 else False

            pred_eeg = best_model_eeg(data).argmax(dim=-1).unfold(0,data[0].shape[1],data[0].shape[1]).detach().cpu().numpy()
            pred_eog = best_model_eog(data).argmax(dim=-1).unfold(0,data[0].shape[1],data[0].shape[1]).detach().cpu().numpy()
            pred_mm = best_model_multimodal(data)["preds"]["combined"].argmax(dim=-1).unfold(0,data[0].shape[1],data[0].shape[1]).detach().cpu().numpy()
            # target_numpy_unfolded = target.unfold(0,data[0].shape[1],data[0].shape[1]).detach().cpu().numpy()
            target_numpy_unfolded = target.detach().cpu().numpy()

            perf_eeg = np.array([f1_score(target_numpy_unfolded[i], pred_eeg[i], average="macro") for i in range(target_numpy_unfolded.shape[0])])
            perf_eog = np.array([f1_score(target_numpy_unfolded[i], pred_eog[i], average="macro") for i in range(target_numpy_unfolded.shape[0])])
            perf_mm = np.array([f1_score(target_numpy_unfolded[i], pred_mm[i], average="macro") for i in range(target_numpy_unfolded.shape[0])])

            #np.equal(tts_unfolded[i], preds_unfolded[i]).sum() / len(tts_unfolded[i])

            router_target = np.concatenate([np.expand_dims(perf_mm, axis=1), np.expand_dims(perf_eeg, axis=1), np.expand_dims(perf_eog, axis=1) ],axis=1)
            router_target = torch.from_numpy(router_target).cuda()
            # router_target = nn.Softmax(dim=1)((router_target.transpose(0,1) - torch.min(router_target,dim=-1)[0]).transpose(0,1))
            router_target = nn.Softmax(dim=1)(router_target)

            # pred, matches = self.get_predictions_time_series_alignment(views, inits)
            output = self.agent.model(data, return_matches=return_matches, return_order=return_order)

            teacher_preds = teacher_preds.to(self.agent.device).flatten(start_dim=0, end_dim=1)  if "kd_label" in self.agent.config and self.agent.config.kd_label else None
            ce_loss = {}
            for k, v in output["preds"].items():
                ce_loss[k] = self.agent.loss(v, router_target, teacher_preds) if teacher_preds else self.agent.loss(v, router_target)

            total_loss = 0
            output_losses = {}
            for i in ce_loss:
                total_loss += self.w_loss[i] * ce_loss[i]
                ce_loss[i] = ce_loss[i].detach().cpu().numpy()
                output_losses.update({"ce_loss_{}".format(i): ce_loss[i]})

            total_loss =  total_loss.detach().cpu().numpy()
            output_losses.update({"total": total_loss})

            for i in output["preds"]:  output["preds"][i] =  output["preds"][i].detach().cpu().numpy()

            return output_losses, output["preds"], router_target
    def sleep_valtest_one_step_reconstruction(self, data, target, inits, teacher_preds=None):


            # pred, matches = self.get_predictions_time_series_alignment(views, inits)
            output = self.agent.model(data)

            output_losses = self.agent.model.module.loss_function(output[0],output[1],output[2],output[3])

            output_losses["total"] =  output_losses["total"].detach().cpu().numpy()

            return output_losses, {}, target

    def get_predictions_time_series(self, views, inits, skip_modality=None):
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
            pred ={"preds":{"combined": torch.zeros(batch * outer, 5).to(this_view.device)}}
            for batch_idx in inits_sum_batch:
                ones_idx = (this_inits[batch_idx] > 0).nonzero(as_tuple=True)[0]
                if (ones_idx[0] + 1 == ones_idx[1]  ): #and ones_idx[0]!=0 and ones_idx[1]!= len(inits[idx])
                    if ones_idx[0] == 0:
                        this_skip_modality = {view: skip_modality[view][batch_idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views} if skip_modality else None
                        pred_split_0 = self.agent.model({view: views[view][batch_idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views}, skip_modality=this_skip_modality)
                    else:
                        this_skip_modality = {view: skip_modality[view][batch_idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views} if skip_modality else None
                        pred_split_0 = self.agent.model({view: views[view][batch_idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views}, skip_modality=this_skip_modality)

                    if ones_idx[1] == len(this_inits[batch_idx]):
                        this_skip_modality = {view: skip_modality[view][batch_idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views} if skip_modality else None
                        pred_split_1 = self.agent.model({view: views[view][batch_idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views}, skip_modality=this_skip_modality)
                    else:
                        this_skip_modality = {view: skip_modality[view][batch_idx, ones_idx[1]:].unsqueeze(dim=0) for view in views} if skip_modality else None
                        pred_split_1 = self.agent.model({view: views[view][batch_idx, ones_idx[1]:].unsqueeze(dim=0) for view in views}, skip_modality=this_skip_modality)

                    pred["preds"]["combined"][batch_idx * outer:(batch_idx + 1) * outer] = torch.cat([pred_split_0["preds"]["combined"], pred_split_1["preds"]["combined"]], dim=0)
                else:
                    this_skip_modality = {view: skip_modality[view][batch_idx].unsqueeze(dim=0) for view in views} if skip_modality else None
                    pred["preds"]["combined"][batch_idx * outer:(batch_idx + 1) * outer] = self.agent.model({view: views[view][batch_idx].unsqueeze(dim=0) for view in views}, skip_modality=this_skip_modality)["preds"]["combined"]

                batch_idx_checked[batch_idx] = False

            this_skip_modality = {view: skip_modality[view][batch_idx_checked] for view in views} if skip_modality else None
            pred["preds"]["combined"][batch_idx_checked.repeat_interleave(outer)] = self.agent.model({view: views[view][batch_idx_checked] for view in views}, skip_modality=this_skip_modality)["preds"]["combined"]

        else:
            pred = self.agent.model(views, skip_modality=skip_modality)

        return pred
    def get_predictions_time_series_onlyskip(self, views, inits, skip_modality=None, **kwargs):

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
