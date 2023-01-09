import einops
import torch
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from graphs.models.custom_unet import *
from graphs.models.bilstm_att import *
from graphs.models.custom_layers.eeg_encoders import *
from graphs.models.attention_models.windowFeature_base import *
from graphs.models.Epilepsy_models.Epilepsy_CNN import *
import os.path
from torchdistill.core.forward_hook import ForwardHookManager

class Sleep_Agent():
    def __init__(self, config):
        self.config = config
        self.steps_no_improve = 0

    def sleep_load(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        print("Loading from file {}".format(file_name))
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_model.load_state_dict(checkpoint["best_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.logs = checkpoint["logs"]
        self.data_loader.load_metrics_ongoing(checkpoint["metrics"])
        print("Metrics have been loaded")
        self.data_loader.weights = self.logs["weights"]
        self.weights = self.data_loader.weights
        self.loss = nn.CrossEntropyLoss(self.weights.to(self.device))
        print("Weights are: ", end="")
        print(self.weights)

        print("Model has loaded successfully")

    def sleep_load_encoder(self):
        encs = []
        for num_enc in range(len(self.config.model.encoders)):
            if self.config.model.encoders[num_enc]["model"] == "TF":
                layers = ["huy_pos_inner", "inner_att", "aggregation_att_contx_inner", "huy_pos_outer", "outer_att"]
                enc = Multi_Transformer(128, inner= 29, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)
            else:
                enc_class = globals()[self.config.model.encoders[num_enc]["model"]]
                args = self.config.model.encoders[num_enc]["args"]
                print(enc_class)
                enc = enc_class(args = args)
                enc = nn.DataParallel(enc, device_ids=[torch.device(i) for i in self.config.gpu_device])

            if self.config.model.encoders[num_enc]["pretrainedEncoder"]["use"]:
                print("Loading encoder from {}".format(self.config.model.encoders[num_enc]["pretrainedEncoder"]["dir"]))
                checkpoint = torch.load(self.config.model.encoders[num_enc]["pretrainedEncoder"]["dir"])
                enc.load_state_dict(checkpoint["encoder_state_dict"])
            encs.append(enc)
        return encs

    def sleep_save(self, file_name="checkpoint.pth.tar"):
            """
            Checkpoint saver
            :param file_name: name of the checkpoint file
            :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
            :return:
            """
            save_dict = {}
            savior = {}
            savior["model_state_dict"] = self.model.state_dict()
            savior["best_model_state_dict"] = self.best_model.state_dict()
            savior["optimizer_state_dict"] = self.optimizer.state_dict()
            savior["logs"] = self.logs
            savior["metrics"] = self.data_loader.metrics

            save_dict.update(savior)

            if self.config.split_method == "patients_folds": file_name = file_name.format(self.config.fold)

            try:
                torch.save(save_dict, file_name)
                second_filename = file_name.split(".")[-3] +"_cp.pth.tar"
                print(second_filename)
                torch.save(save_dict, second_filename)
                if self.config.verbose:
                    print("Models has saved successfully in {}".format(file_name))
            except:
                raise Exception("Problem in model saving")

    def sleep_train_one_epoch(self):
            """
            One epoch of training
            :return:
            """
            self.model.train()
            for enc in range(len(self.config.encoder_models)):
                if self.config.freeze_encoders[i]:
                    if hasattr(self.model.module,"enc_{}".format(i)):
                        for p in getattr(self.model.module,"enc_{}".format(enc)).parameters():
                            p.requires_grad = False

            batch_loss = 0
            tts, preds = [], []
            pbar = tqdm(enumerate(self.data_loader.train_loader), desc = "Training", leave=False, disable=self.config.tdqm_disable)
            for batch_idx, (data, target, _, idxs) in pbar: #tqdm(enumerate(self.data_loader.train_loader), "Training", leave=False, disable=self.config.tdqm_disable):
                views = [data[i].float().to(self.device) for i in range(len(data))]
                target = target.to(self.device).flatten()
                self.optimizer.zero_grad()
                pred = self.model(views)
                loss = self.loss(pred, target)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                loss.backward()
                #update progress bar
                pbar.set_description("Training batch {0:d}/{1:d} with loss {2:.5f}".format(batch_idx,len(self.data_loader.train_loader),loss.item()))
                pbar.refresh()

                batch_loss += loss
                tts.append(target)
                preds.append(pred)
                self.optimizer.step()
                self.scheduler.step()
            # self.model.module.enc_0.dy_conv_0.update_temperature()
            # self.model.module.enc_0.dy_conv_1.update_temperature()
            # self.model.module.enc_0.dy_conv_3.update_temperature()
            tts = torch.cat(tts).cpu().numpy()
            preds = torch.cat(preds).argmax(axis=1).cpu().numpy()
            return batch_loss / len(self.data_loader.train_loader), np.equal(tts, preds).sum() / len(tts), f1_score(preds, tts, average="macro"), cohen_kappa_score(preds, tts, average="macro")

    def get_predictions_time_series(self, views, inits):
        """
        This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
        :param views: List of tensors, data views/modalities
        :param inits: Tensor indicating with value one, when there incontinuities.
        :return: predictions of the self.model on the batch
        """
        inits_sum = (inits.sum(dim=1) > 1).nonzero(as_tuple=True)[0]
        if len(inits_sum) > 0:
            batch = views[0].shape[0]
            outer = views[0].shape[1]
            batch_idx_checked = torch.ones(batch, dtype=torch.bool)
            pred = torch.zeros(batch * outer, self.config.num_classes).cuda()
            for idx in inits_sum:
                if inits[idx].sum() > 1:
                    ones_idx = (inits[idx] > 0).nonzero(as_tuple=True)[0]
                    if (ones_idx[0] + 1 == ones_idx[1]  ): #and ones_idx[0]!=0 and ones_idx[1]!= len(inits[idx])
                        if ones_idx[0] == 0:
                            pred_split_0 = self.model([view[idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                        else:
                            pred_split_0 = self.model([view[idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views])
                        if ones_idx[1] == len(inits[idx]):
                            pred_split_1 = self.model(
                                [view[idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                        else:
                            pred_split_1 = self.model([view[idx, ones_idx[1]:].unsqueeze(dim=0) for view in views])

                        pred[idx * outer:(idx + 1) * outer] = torch.cat([pred_split_0, pred_split_1], dim=0)
                        batch_idx_checked[idx] = False
                    else:
                        pred[idx * outer:(idx + 1) * outer] = self.model([view[idx].unsqueeze(dim=0) for view in views])

            pred[batch_idx_checked.repeat_interleave(outer)] = self.model([view[batch_idx_checked] for view in views])

        else:
            pred = self.model(views)

        return pred
    def get_predictions_time_series_blip(self, views, inits):
        """
        This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
        :param views: List of tensors, data views/modalities
        :param inits: Tensor indicating with value one, when there incontinuities.
        :return: predictions of the self.model on the batch
        """
        inits_sum = (inits.sum(dim=1) > 1).nonzero(as_tuple=True)[0]
        if len(inits_sum) > 0:
            batch = views[0].shape[0]
            outer = views[0].shape[1]
            batch_idx_checked = torch.ones(batch, dtype=torch.bool)
            pred = torch.zeros(batch * outer, self.config.num_classes).cuda()
            matches = torch.zeros(batch, outer, outer).cuda()
            for idx in inits_sum:
                if inits[idx].sum() > 1:
                    ones_idx = (inits[idx] > 0).nonzero(as_tuple=True)[0]
                    if (ones_idx[0] + 1 == ones_idx[1]  ): #and ones_idx[0]!=0 and ones_idx[1]!= len(inits[idx])
                        if ones_idx[0] == 0:
                            pred_split_0, matches_0 = self.model([view[idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                        else:
                            pred_split_0, matches_0= self.model([view[idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views])
                        if ones_idx[1] == len(inits[idx]):
                            pred_split_1, matches_1 = self.model([view[idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                        else:
                            pred_split_1, matches_1 = self.model([view[idx, ones_idx[1]:].unsqueeze(dim=0) for view in views])

                        pred[idx * outer:(idx + 1) * outer] = torch.cat([pred_split_0, pred_split_1], dim=0)
                        print(matches_0.shape)
                        print(matches_1.shape)
                        matches[idx * outer:(idx + 1) * outer] = torch.cat([matches_0, matches_1], dim=0)
                        batch_idx_checked[idx] = False
                    else:
                        pred[idx * outer:(idx + 1) * outer], matches[idx * outer:(idx + 1) * outer] = self.model([view[idx].unsqueeze(dim=0) for view in views])

            pred[batch_idx_checked.repeat_interleave(outer)], matches[batch_idx_checked.repeat_interleave(outer)] = self.model([view[batch_idx_checked] for view in views])

        else:
            pred, matches = self.model(views)

        return pred
    def get_predictions_time_series_teacher(self, views, inits):
        """
        This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
        :param views: List of tensors, data views/modalities
        :param inits: Tensor indicating with value one, when there incontinuities.
        :return: predictions of the self.model on the batch
        """
        inits_sum = (inits.sum(dim=1) > 1).nonzero(as_tuple=True)[0]
        if len(inits_sum) > 0:
            batch = views[0].shape[0]
            outer = views[0].shape[1]
            batch_idx_checked = torch.ones(batch, dtype=torch.bool)
            pred = torch.zeros(batch * outer, self.config.num_classes).cuda()
            t_pred = torch.zeros(batch * outer, self.config.num_classes).cuda()
            for idx in inits_sum:
                if inits[idx].sum() > 1:
                    ones_idx = (inits[idx] > 0).nonzero(as_tuple=True)[0]
                    if (ones_idx[0] + 1 == ones_idx[1]  ): #and ones_idx[0]!=0 and ones_idx[1]!= len(inits[idx])
                        if ones_idx[0] == 0:
                            pred_split_0, t_pred_split_0 = self.model([view[idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                        else:
                            pred_split_0, t_pred_split_0 = self.model([view[idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views])
                        if ones_idx[1] == len(inits[idx]):
                            pred_split_1, t_pred_split_1 = self.model([view[idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                        else:
                            pred_split_1, t_pred_split_1 = self.model([view[idx, ones_idx[1]:].unsqueeze(dim=0) for view in views])

                        pred[idx * outer:(idx + 1) * outer] = torch.cat([pred_split_0, pred_split_1], dim=0)
                        t_pred[idx * outer:(idx + 1) * outer] = torch.cat([t_pred_split_0, t_pred_split_1], dim=0)
                        batch_idx_checked[idx] = False
                    else:
                        pred[idx * outer:(idx + 1) * outer],t_pred[idx * outer:(idx + 1) * outer] = self.model([view[idx].unsqueeze(dim=0) for view in views])

            pred[batch_idx_checked.repeat_interleave(outer)], t_pred[batch_idx_checked.repeat_interleave(outer)] = self.model([view[batch_idx_checked] for view in views])

        else:
            pred, t_pred = self.model(views)

        return pred, t_pred

    def sleep_train_one_step(self, data, target, inits, teacher_preds=None, current_epoch=0):

            views = [data[i].float().to(self.device) for i in range(len(data))]

            #If not softlabels
            #target = target.flatten()

            if "kd_label" in self.config and self.config.kd_label: teacher_preds = teacher_preds.to(self.device).flatten(start_dim=0,end_dim=1)

            self.optimizer.zero_grad()
            # pred = self.get_predictions_time_series(views, inits)
            pred = self.model(views, inits)
            if "kd_label" in self.config and self.config.kd_label:
                loss = self.loss(pred, target, teacher_preds)
            else:
                loss = self.loss(pred, target)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()

            return [loss], pred
    def sleep_train_one_step_blip(self, data, target, inits, teacher_preds=None, current_epoch=0):

            views = [data[i].float().to(self.device) for i in range(len(data))]

            #If not softlabels
            #target = target.flatten()

            if "kd_label" in self.config and self.config.kd_label: teacher_preds = teacher_preds.to(self.device).flatten(start_dim=0,end_dim=1)

            self.optimizer.zero_grad()
            # pred, matches = self.get_predictions_time_series_blip(views, inits)
            pred, matches = self.model(views, return_matches=True)

            if "kd_label" in self.config and self.config.kd_label:
                ce_loss = self.loss(pred, target, teacher_preds)
            else:
                ce_loss = self.loss(pred, target)

            matches = matches.flatten(start_dim=0, end_dim=1)
            if "blip_loss" in self.config:
                blip_target = self.blip_target[:views[0].shape[0], :views[0].shape[1], :views[0].shape[1]].flatten(start_dim=0, end_dim=1)

            else:
                blip_target = self.blip_target[:views[0].shape[0], :views[0].shape[1]].flatten(start_dim=0, end_dim=1)
            blip_loss = self.blip_loss(matches, blip_target)

            if "blip_weights" in self.config.model.blip:
                w_supervised_loss = self.config.model.blip.blip_weights["supervised_loss"]
                w_alignments_loss = self.config.model.blip.blip_weights["alignment_loss"]

            total_loss = w_supervised_loss*ce_loss + w_alignments_loss*blip_loss

            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step(current_epoch)

            return [total_loss, ce_loss, blip_loss], pred
    def sleep_train_one_step_blip_twoloss(self, data, target, inits, teacher_preds=None, current_epoch=0):

            views = [data[i].float().to(self.device) for i in range(len(data))]

            #If not softlabels
            #target = target.flatten()

            if "kd_label" in self.config and self.config.kd_label: teacher_preds = teacher_preds.to(self.device).flatten(start_dim=0,end_dim=1)

            self.optimizer.zero_grad()
            # pred, matches = self.get_predictions_time_series_blip(views, inits)
            pred, matches, inter_pred, order_pred = self.model(views, return_matches=True, return_reps=True)

            if "kd_label" in self.config and self.config.kd_label:
                ce_loss = self.loss(pred, target, teacher_preds)
            else:
                ce_loss = self.loss(pred, target)

            matches = matches.flatten(start_dim=0, end_dim=1)
            if "blip_loss" in self.config:
                blip_target = self.blip_target[:views[0].shape[0], :views[0].shape[1], :views[0].shape[1]].flatten(start_dim=0, end_dim=1)
            else:
                blip_target = self.blip_target[:views[0].shape[0], :views[0].shape[1]].flatten(start_dim=0, end_dim=1)

            alignment_loss = self.blip_loss(matches, blip_target)
            order_target = torch.zeros([32, 19]).cuda() != 0

            unfolded_target = einops.rearrange(target," (b outer) -> b outer", b=32, outer=21)
            unfolded_target = unfolded_target.unfold(1,3,1)
            d1 = unfolded_target[:, :, 0] == unfolded_target[:, :, 1]
            d2 = unfolded_target[:, :, 2] == unfolded_target[:, :, 1]
            order_target[d1] = d2[d1] == True
            order_target = order_target.flatten().long()
            order_loss = self.blip_loss(order_pred, order_target)

            if "blip_weights" in self.config.model.blip:
                w_supervised_loss = self.config.model.blip.blip_weights["supervised_loss"]
                w_alignments_loss = self.config.model.blip.blip_weights["alignment_loss"]
                w_order_loss = self.config.model.blip.blip_weights["order_loss"]

            total_loss = w_supervised_loss*ce_loss + w_alignments_loss*alignment_loss + w_order_loss*order_loss

            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step(current_epoch)

            return [total_loss, ce_loss, alignment_loss, order_loss], pred

    def sleep_train_one_step_sparse(self, data, target, inits, current_epoch=0):

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

    def sleep_train_one_step_adv(self, data, target, inits, current_epoch=0):
            torch.autograd.set_detect_anomaly(True)

            self.optimizer.zero_grad()
            views = [data[i].float().to(self.device) for i in range(len(data))]
            for i in views: i.requires_grad = True
            target = target.to(self.device).flatten()
            # pred = self.model(views)
            pred = self.get_predictions_time_series(views, inits)
            loss = self.loss(pred, target)
            loss.backward(retain_graph=True)
            views_plus = [view + self.config.adv_epsilon * (view.grad).sign() for view in views]
            # pred_plus = self.model(views_plus)
            # print("There is a difference here, check it")
            self.optimizer.zero_grad()
            pred_plus = self.get_predictions_time_series(views_plus, inits)
            loss_plus = self.loss(pred_plus, target)
            loss_total = loss_plus + loss
            loss_total.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step(current_epoch)

            return [loss_total, loss, loss_plus], pred
    def sleep_train_one_step_blip_adv(self, data, target, inits, teacher_preds=None, current_epoch=0):
            torch.autograd.set_detect_anomaly(True)

            self.optimizer.zero_grad()
            views = [data[i].float().to(self.device) for i in range(len(data))]
            for i in views: i.requires_grad = True
            target = target.to(self.device).flatten()
            # pred = self.model(views)
            # pred = self.get_predictions_time_series(views, inits)
            pred, matches = self.model(views, return_matches=True)

            if "kd_label" in self.config and self.config.kd_label:
                loss_ce = self.loss(pred, target, teacher_preds)
            else:
                loss_ce = self.loss(pred, target)

            # i_diagonal = views[0].shape[1] - 1
            # diagonal_vals = torch.ones(i_diagonal, dtype=torch.long)
            # diag_matrix = (torch.diagflat(diagonal_vals, offset=1) + torch.diagflat(diagonal_vals, offset=-1)) * 0.1
            # diagonal_vals = torch.ones(views[0].shape[1], dtype=torch.long) * 0.8
            # diag_matrix += torch.diagflat(diagonal_vals)
            # blip_target = diag_matrix
            # blip_target = blip_target.unsqueeze(dim=0).repeat(views[0].shape[0], 1, 1).flatten(start_dim=0, end_dim=1).cuda()


            # blip_target = torch.eye(n=views[0].shape[1])
            # blip_target = blip_target.unsqueeze(dim=0).repeat(views[0].shape[0], 1, 1).flatten(start_dim=0, end_dim=1).cuda()
            matches = matches.flatten(start_dim=0, end_dim=1)

            if "blip_loss" in self.config:
                blip_target = self.blip_target[:views[0].shape[0], :views[0].shape[1], :views[0].shape[1]].flatten(start_dim=0, end_dim=1)

            else:
                blip_target = self.blip_target[:views[0].shape[0], :views[0].shape[1]].flatten(start_dim=0, end_dim=1)

            loss_blip = self.blip_loss(matches, blip_target)
            loss_ce = 4*loss_ce
            loss = loss_ce + loss_blip
            loss.backward(retain_graph=True)

            views_plus = [view + self.config.adv_epsilon * (view.grad).sign() for view in views]
            # pred_plus = self.model(views_plus)
            # print("There is a difference here, check it")
            self.optimizer.zero_grad()
            pred_plus, matches_plus = self.model(views_plus, return_matches=True)
            loss_plus_ce = self.loss(pred_plus, target)
            loss_plus_ce = 4*loss_plus_ce
            loss_plus_blip = self.blip_loss(matches_plus.flatten(start_dim=0, end_dim=1), blip_target)
            loss_plus = loss_plus_ce + loss_plus_blip

            # pred_plus = self.get_predictions_time_series(views_plus, inits)
            loss_total = loss_plus + loss
            loss_total.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step(current_epoch)

            del pred_plus
            del matches_plus
            del matches

            return [loss_plus, loss, loss_plus_ce, loss_plus_blip, loss_ce, loss_blip], pred

    def monitoring(self, a, b):
        [train_loss, train_acc, train_f1, train_k] = a
        [val_loss, val_acc, val_f1, val_k, val_perclassf1] = b
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
        self.logs["val_logs"][self.logs["current_step"]] = {"val_loss":val_loss,"val_k":val_k,"val_f1":val_f1,"val_acc":val_acc,"val_perclassf1":val_perclassf1}
        self.logs["train_logs"][self.logs["current_step"]] = {"train_loss":train_loss[0],"train_k":train_k,"train_f1":train_f1,"train_acc":train_acc,"validate_every":self.config.validate_every,"batch_size":self.config.batch_size, "learning_rate": lr}
        if len(train_loss)>1:
            for i in range(1, len(train_loss)):
                self.logs["train_logs"][self.logs["current_step"]]["additional_loss_{}".format(i)] = train_loss[i]
        early_stop, not_saved, step = False, True, int(self.logs["current_step"] / self.config.validate_every)

        if self.config.verbose:
            print("Epoch {0:d}, N: {1:d}, lr: {2:.6f} Validation loss: {3:.6f}, accuracy: {4:.2f}% f1 :{5:.4f},  :{6:.4f}  Training loss: {7:.6f}, accuracy: {8:.2f}% f1 :{9:.4f}, k :{10:.4f},".format(
                    self.logs["current_epoch"], self.logs["current_step"] * self.config.batch_size * self.config.seq_legth[0], lr, self.logs["val_logs"][self.logs["current_step"]]["val_loss"], self.logs["val_logs"][self.logs["current_step"]]["val_acc"] * 100,
                    self.logs["val_logs"][self.logs["current_step"]]["val_f1"], self.logs["val_logs"][self.logs["current_step"]]["val_k"], self.logs["train_logs"][self.logs["current_step"]]["train_loss"], self.logs["train_logs"][self.logs["current_step"]]["train_acc"] * 100,
                    self.logs["train_logs"][self.logs["current_step"]]["train_f1"],train_k))
        if (val_loss < self.logs["best_logs"]["val_loss"]):
            self.logs["best_logs"] = {"step":self.logs["current_step"], "val_loss":self.logs["val_logs"][self.logs["current_step"]]["val_loss"], "val_acc":self.logs["val_logs"][self.logs["current_step"]]["val_acc"], "val_f1":self.logs["val_logs"][self.logs["current_step"]]["val_f1"],
                                      "val_k":self.logs["val_logs"][self.logs["current_step"]]["val_k"],  "val_perclassf1":self.logs["val_logs"][self.logs["current_step"]]["val_perclassf1"]}
            print("we have a new best at epoch {0:d} step {1:d} with validation loss: {2:.6f} accuracy: {3:.2f}%, f1: {4:.4f}, k: {5:.4f},  f1_per_class :{6:40}".format(self.logs["current_epoch"], step, val_loss, val_acc * 100, val_f1, val_k, "{}".format(list(val_perclassf1))))
            self.best_model.load_state_dict(self.model.state_dict())
            if self.config.rec_test:
                test_loss, test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens = self.sleep_test()
                print("Test loss: {0:.6}, accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, sens:{4:.4f}, spec:{5:.4f}, f1_per_class :{6:40}".format(test_loss,
                                                                                                         test_acc * 100,
                                                                                                         test_f1,
                                                                                                         test_k, test_spec, test_sens,
                                                                                                         "{}".format(list(test_perclass_f1))))
                self.logs["test_logs"][self.logs["current_step"]] = {"test_loss": test_loss, "test_k": test_k,
                                                                    "test_f1": test_f1, "test_acc": test_acc, "test_spec": test_spec, "test_sens": test_conf,
                                                                    "test_acc": test_conf, "test_auc":test_auc, "test_perclass_f1": list(test_perclass_f1)}
            self.sleep_save(self.config.model.save_dir)
            self.save_encoder()
            not_saved = False
            self.logs["steps_no_improve"] = 0
        else:
            if self.config.rec_test and self.config.test_on_tops:
                test_loss, test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens = self.sleep_test()
                print(
                    "Test loss: {0:.6}, accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, sens:{4:.4f}, spec:{5:.4f}, f1_per_class :{6:40}".format(test_loss,
                                                                                                         test_acc * 100,
                                                                                                         test_f1,
                                                                                                         test_k, test_spec, test_sens,
                                                                                                         "{}".format(list(test_perclass_f1))))
                self.logs["test_logs"][self.logs["current_step"]] = {"test_loss": test_loss, "test_k": test_k,
                                                                    "test_f1": test_f1, "test_acc": test_acc, "test_spec": test_spec, "test_sens": test_conf,
                                                                    "test_acc": test_conf, "test_auc":test_auc, "test_perclass_f1": list(test_perclass_f1)}
            self.logs["steps_no_improve"] += 1
            print("Current steps with no improvement {}".format(self.logs["steps_no_improve"]))
        if ((self.logs["current_step"] // self.config.validate_every) % self.config.save_every == 0 and not_saved):
            self.sleep_save(self.config.model.save_dir)
            self.save_encoder()

        if self.config.verbose:
            print("This epoch took {} seconds".format(time.time() - self.start))
        if self.logs["current_step"] == self.config.n_steps_stop_after * self.config.validate_every:
            self.logs["steps_no_improve"] = 0
        if self.logs["current_step"] > self.config.n_steps_stop_after * self.config.validate_every and self.logs["steps_no_improve"] >= self.config.n_steps_stop:
            print('Early stopping!')
            early_stop = True

        return early_stop

    def sleep_train_step(self):
        self.model.train()

        for enc in range(len(self.config.model. encoders)):
            if "freeze_encoder" in self.config.model.encoders[enc] and self.config.model.encoders[enc]["freeze_encoder"]:
                if hasattr(self.model.module, "enc_{}".format(enc)):
                    print("Freezing encoder enc_{}".format(enc))
                    for p in getattr(self.model.module, "enc_{}".format(enc)).parameters():
                        p.requires_grad = False

        tts, preds, batch_loss, datapoints_sum, early_stop = [], [], [], 0, False
        self.start = time.time()
        for self.logs["current_epoch"] in range(self.logs["current_epoch"], self.config.max_epoch):
            pbar = tqdm(enumerate(self.data_loader.train_loader), desc="Training", leave=None, disable=self.config.tdqm_disable, position=0)

            for batch_idx, batch in pbar:
                data, target, inits = batch[0], batch[1], batch[2]
                # shuffle
                if "random_shuffle_data" in self.config and self.config.random_shuffle_data:
                    perms = torch.randperm(data[0].shape[1])
                    data = [view[:, perms] for view in data]
                    target = target[:, perms].flatten()
                    inits = inits[:, perms]

                if "random_shuffle_data_batch" in self.config and self.config.random_shuffle_data_batch:
                    perms = torch.randperm(data[0].shape[0] * data[0].shape[1])
                    d_shape = data[0].shape
                    data = [einops.rearrange(einops.rearrange(view, "batch seq b c d -> (batch seq) b c d")[perms],
                                              "(batch seq) b c d -> batch seq b c d", batch=d_shape[0], seq=d_shape[1])
                             for view in data]
                    target = target.flatten()[perms]
                    inits = einops.rearrange(einops.rearrange(inits, "batch seq -> (batch seq)")[perms],
                                             "(batch seq) -> batch seq", batch=d_shape[0], seq=d_shape[1])

                if "kd_label" in self.config and self.config.kd_label:
                    teacher_preds = batch[4]
                else:
                    teacher_preds = None

                if "softlabels" in self.config and self.config.softlabels:
                    target = target.float()
                else:
                    target = target.to(self.device).flatten(start_dim=0, end_dim=1).long()
                    if len(target.shape) > 1:
                        target = target.argmax(dim=1)

                self.model.train()
                if "use_adversarial" in self.config and self.config.use_adversarial:
                    loss, pred = self.sleep_train_one_step_adv(data, target, inits)
                else:
                    if "sparse_loss" in self.config and self.config.sparse_loss:
                        loss, pred = self.sleep_train_one_step_sparse(data, target, inits)
                    else:
                        loss, pred = self.sleep_train_one_step(data, target, inits, teacher_preds)


                batch_loss.append([i.cpu().detach().numpy() for i in loss])
                del loss
                datapoints_sum+=1
                tts.append(target)
                preds.append(pred)
                pbar_message = "Training batch {0:d}/{1:d} with".format(batch_idx, len(self.data_loader.train_loader)-1)
                for i, v in enumerate(np.array(batch_loss).mean(axis=0)):
                    pbar_message += "loss: {:.3f} ".format(v) if i==0 else "add_loss_{}: {:.3f} ".format(i,v)
                pbar.set_description(pbar_message)
                pbar.refresh()
                if self.logs["current_step"] % self.config.validate_every == 0 and batch_idx!=0 and  self.logs["current_step"] % self.config.validate_every >= self.config.validate_after :
                    # print("We are in validation")

                    del data, target, pred

                    if "softlabels" in self.config and self.config.softlabels:
                        tts = torch.cat(tts).argmax(dim=1).cpu().numpy().flatten()
                    else:
                        tts = torch.cat(tts).cpu().numpy().flatten()
                    preds = torch.cat(preds).argmax(axis=1).cpu().numpy()
                    train_loss, train_acc, train_f1, train_k = np.array(batch_loss).mean(axis=0), np.equal(tts, preds).sum() / len(tts), f1_score(preds,tts,average="macro"), cohen_kappa_score(preds, tts)
                    batch_loss, datapoints_sum, tts, preds = [], 0, [], []
                    val_loss, val_acc, val_f1, val_k, val_perclassf1 = self.sleep_validate()
                    # self.model.train()
                    early_stop = self.monitoring([train_loss, train_acc, train_f1, train_k],[val_loss, val_acc, val_f1, val_k, val_perclassf1])
                    if early_stop: break
                    self.start = time.time()

                self.logs["current_step"] += 1
            if early_stop: break

        return np.array(batch_loss).mean(axis=0), np.equal(tts, preds).sum() / len(tts), f1_score(preds,tts, average="macro"), cohen_kappa_score(preds, tts)

    def sleep_train_step_blip(self):
        self.model.train()

        for enc in range(len(self.config.model. encoders)):
            if "freeze_encoder" in self.config.model.encoders[enc] and self.config.model.encoders[enc]["freeze_encoder"]:
                if hasattr(self.model.module, "enc_{}".format(enc)):
                    print("Freezing encoder enc_{}".format(enc))
                    for p in getattr(self.model.module, "enc_{}".format(enc)).parameters():
                        p.requires_grad = False

        tts, preds, batch_loss, datapoints_sum, early_stop = [], [], [], 0, False
        self.start = time.time()

        number_batches = len(self.data_loader.train_loader)
        for self.logs["current_epoch"] in range(self.logs["current_epoch"], self.config.max_epoch):
            pbar = tqdm(enumerate(self.data_loader.train_loader), desc="Training", leave=None, disable=self.config.tdqm_disable, position=0)

            for batch_idx, batch in pbar:
                data, target, inits = batch[0], batch[1], batch[2]
                # shuffle
                if "random_shuffle_data" in self.config and self.config.random_shuffle_data:
                    perms = torch.randperm(data[0].shape[1])
                    data = [view[:, perms] for view in data]
                    target = target[:, perms].flatten()
                    inits = inits[:, perms]

                if "random_shuffle_data_batch" in self.config and self.config.random_shuffle_data_batch:
                    perms = torch.randperm(data[0].shape[0] * data[0].shape[1])
                    d_shape = data[0].shape
                    data = [einops.rearrange(einops.rearrange(view, "batch seq b c d -> (batch seq) b c d")[perms],
                                              "(batch seq) b c d -> batch seq b c d", batch=d_shape[0], seq=d_shape[1])
                             for view in data]
                    target = target.flatten()[perms]
                    inits = einops.rearrange(einops.rearrange(inits, "batch seq -> (batch seq)")[perms],
                                             "(batch seq) -> batch seq", batch=d_shape[0], seq=d_shape[1])

                if "kd_label" in self.config and self.config.kd_label:
                    teacher_preds = batch[4]
                else:
                    teacher_preds = None
                #
                if "softlabels" in self.config and self.config.softlabels:
                    target = target.float()
                else:
                    target = target.to(self.device).flatten(start_dim=0, end_dim=1).long()
                    if len(target.shape) > 1:
                        target = target.argmax(dim=1)
                #
                if "use_adversarial" in self.config and self.config.use_adversarial:
                    loss, pred = self.sleep_train_one_step_blip_adv(data, target, inits, current_epoch=(self.logs["current_epoch"]+batch_idx)/number_batches)
                else:
                    if "sparse_loss" in self.config and self.config.sparse_loss:
                        loss, pred = self.sleep_train_one_step_sparse(data, target, inits, current_epoch=(self.logs["current_epoch"]+batch_idx)/number_batches)
                    else:
                        loss, pred = self.sleep_train_one_step_blip(data, target, inits, teacher_preds, current_epoch=(self.logs["current_epoch"]+batch_idx)/number_batches)
                batch_loss.append([i.cpu().detach().numpy() for i in loss])
                del loss
                datapoints_sum+=1
                tts.append(target.cpu().detach())
                preds.append(pred.cpu().detach())
                del pred
                pbar_message = "Training batch {0:d}/{1:d} with".format(batch_idx, len(self.data_loader.train_loader)-1)
                for i, v in enumerate(np.array(batch_loss).mean(axis=0)):
                    pbar_message += "loss: {:.3f} ".format(v) if i==0 else "add_loss_{}: {:.3f} ".format(i,v)
                pbar.set_description(pbar_message)
                pbar.refresh()
                if self.logs["current_step"] % self.config.validate_every == 0 and batch_idx!=0 and  self.logs["current_step"] % self.config.validate_every >= self.config.validate_after :
                    # print("We are in validation")

                    del data, target

                    if "softlabels" in self.config and self.config.softlabels:
                        tts = torch.cat(tts).argmax(dim=1).cpu().numpy().flatten()
                    else:
                        tts = torch.cat(tts).cpu().numpy().flatten()
                    preds = torch.cat(preds).argmax(axis=1).cpu().numpy()
                    # batch_loss = np.array(batch_loss).mean(axis=0)
                    # print(batch_loss.shape)
                    train_loss, train_acc, train_f1, train_k = np.array(batch_loss).mean(axis=0), np.equal(tts, preds).sum() / len(tts), f1_score(preds,tts,average="macro"), cohen_kappa_score(preds, tts)
                    batch_loss, datapoints_sum, tts, preds = [], 0, [], []
                    val_loss, val_acc, val_f1, val_k, val_perclassf1 = self.sleep_validate()
                    # self.model.train()
                    early_stop = self.monitoring([train_loss, train_acc, train_f1, train_k],[val_loss, val_acc, val_f1, val_k, val_perclassf1])
                    if early_stop: break
                    self.start = time.time()

                self.logs["current_step"] += 1
            if early_stop: break

        return np.array(batch_loss).mean(axis=0), np.equal(tts, preds).sum() / len(tts), f1_score(preds,tts, average="macro"), cohen_kappa_score(preds, tts)

    def sleep_train_step_blip_twoloss(self):
        self.model.train()

        for enc in range(len(self.config.model. encoders)):
            if "freeze_encoder" in self.config.model.encoders[enc] and self.config.model.encoders[enc]["freeze_encoder"]:
                if hasattr(self.model.module, "enc_{}".format(enc)):
                    print("Freezing encoder enc_{}".format(enc))
                    for p in getattr(self.model.module, "enc_{}".format(enc)).parameters():
                        p.requires_grad = False

        tts, preds, batch_loss, datapoints_sum, early_stop = [], [], [], 0, False
        self.start = time.time()

        number_batches = len(self.data_loader.train_loader)
        for self.logs["current_epoch"] in range(self.logs["current_epoch"], self.config.max_epoch):
            pbar = tqdm(enumerate(self.data_loader.train_loader), desc="Training", leave=None, disable=self.config.tdqm_disable, position=0)

            for batch_idx, batch in pbar:
                data, target, inits = batch[0], batch[1], batch[2]
                # shuffle
                if "random_shuffle_data" in self.config and self.config.random_shuffle_data:
                    perms = torch.randperm(data[0].shape[1])
                    data = [view[:, perms] for view in data]
                    target = target[:, perms].flatten()
                    inits = inits[:, perms]

                if "random_shuffle_data_batch" in self.config and self.config.random_shuffle_data_batch:
                    perms = torch.randperm(data[0].shape[0] * data[0].shape[1])
                    d_shape = data[0].shape
                    data = [einops.rearrange(einops.rearrange(view, "batch seq b c d -> (batch seq) b c d")[perms],
                                              "(batch seq) b c d -> batch seq b c d", batch=d_shape[0], seq=d_shape[1])
                             for view in data]
                    target = target.flatten()[perms]
                    inits = einops.rearrange(einops.rearrange(inits, "batch seq -> (batch seq)")[perms],
                                             "(batch seq) -> batch seq", batch=d_shape[0], seq=d_shape[1])

                if "kd_label" in self.config and self.config.kd_label:
                    teacher_preds = batch[4]
                else:
                    teacher_preds = None
                #
                if "softlabels" in self.config and self.config.softlabels:
                    target = target.float()
                else:
                    target = target.to(self.device).flatten(start_dim=0, end_dim=1).long()
                    if len(target.shape) > 1:
                        target = target.argmax(dim=1)
                #
                if "use_adversarial" in self.config and self.config.use_adversarial:
                    loss, pred = self.sleep_train_one_step_blip_twoloss_adv(data, target, inits, current_epoch=(self.logs["current_epoch"]+batch_idx)/number_batches)
                else:
                    if "sparse_loss" in self.config and self.config.sparse_loss:
                        loss, pred = self.sleep_train_one_step_sparse(data, target, inits, current_epoch=(self.logs["current_epoch"]+batch_idx)/number_batches)
                    else:
                        loss, pred = self.sleep_train_one_step_blip_twoloss(data, target, inits, teacher_preds, current_epoch=(self.logs["current_epoch"]+batch_idx)/number_batches)
                batch_loss.append([i.cpu().detach().numpy() for i in loss])
                del loss
                datapoints_sum+=1
                tts.append(target.cpu().detach())
                preds.append(pred.cpu().detach())
                del pred
                pbar_message = "Training batch {0:d}/{1:d} with".format(batch_idx, len(self.data_loader.train_loader)-1)
                for i, v in enumerate(np.array(batch_loss).mean(axis=0)):
                    pbar_message += "loss: {:.3f} ".format(v) if i==0 else "add_loss_{}: {:.3f} ".format(i,v)
                pbar.set_description(pbar_message)
                pbar.refresh()
                if self.logs["current_step"] % self.config.validate_every == 0 and batch_idx!=0 and  self.logs["current_step"] % self.config.validate_every >= self.config.validate_after :
                    # print("We are in validation")

                    del data, target

                    if "softlabels" in self.config and self.config.softlabels:
                        tts = torch.cat(tts).argmax(dim=1).cpu().numpy().flatten()
                    else:
                        tts = torch.cat(tts).cpu().numpy().flatten()
                    preds = torch.cat(preds).argmax(axis=1).cpu().numpy()
                    # batch_loss = np.array(batch_loss).mean(axis=0)
                    # print(batch_loss.shape)
                    train_loss, train_acc, train_f1, train_k = np.array(batch_loss).mean(axis=0), np.equal(tts, preds).sum() / len(tts), f1_score(preds,tts,average="macro"), cohen_kappa_score(preds, tts)
                    batch_loss, datapoints_sum, tts, preds = [], 0, [], []
                    val_loss, val_acc, val_f1, val_k, val_perclassf1 = self.sleep_validate()
                    # self.model.train()
                    early_stop = self.monitoring([train_loss, train_acc, train_f1, train_k],[val_loss, val_acc, val_f1, val_k, val_perclassf1])
                    if early_stop: break
                    self.start = time.time()

                self.logs["current_step"] += 1
            if early_stop: break

        return np.array(batch_loss).mean(axis=0), np.equal(tts, preds).sum() / len(tts), f1_score(preds,tts, average="macro"), cohen_kappa_score(preds, tts)

    def sleep_test_multi(self):
        for i in range(len(self.models)):
            self.models[i].eval()
        tts = []
        preds = []
        with torch.no_grad():
            for batch_idx, (data, target, init, _) in tqdm(enumerate(self.data_loader.test_loader),"Test",leave=False, disable=self.config.tdqm_disable):
                views = [data[i].float().to(self.device) for i in range(len(data))]
                pred = []
                for i in range(len(self.models)):
                    pred.append(self.models[i](views)*self.vote_weight[i])
                pred = torch.stack(pred, axis=-1)
                pred = pred.mean(axis=-1)
                tts.append(target)
                preds.append(pred)
            tts = torch.cat(tts).cpu().numpy()
            preds = torch.cat(preds).cpu().numpy()
            for w_idx in range(int(self.config.post_proc_step/2 +1 ),len(preds)- int(self.config.post_proc_step/2 +1 )):
                for n_class in range(len(preds[0])):
                    preds[w_idx, n_class] = preds[w_idx-int(self.config.post_proc_step/2):w_idx+int(self.config.post_proc_step/2), n_class].sum()/self.config.post_proc_step
            preds = preds.argmax(axis=1)
            test_acc = np.equal(tts, preds).sum() / len(tts)
            test_f1 = f1_score(preds, tts, average="macro")
            test_k = cohen_kappa_score(tts, preds)
            test_auc = roc_auc_score(tts, preds)
            test_conf = confusion_matrix(tts, preds)
        return test_acc, test_f1, test_k, test_auc, test_conf

    def sleep_plot_losses(self):
        train_loss = np.array([self.logs["train"][i]["train_loss"] for i in self.logs["train_self.logs"]])
        val_loss = np.array([self.logs["val"][i]["val_loss"] for i in self.logs["val_self.logs"]])
        steps = np.array([i / self.logs["train"][i]["validate_every"] for i in self.logs["train_self.logs"]]) - 1

        plt.figure()
        plt.plot(steps, train_loss, label="Train")
        plt.plot(steps, val_loss, label="Valid")

        best_step = self.logs["best_self.logs"]["step"] / self.logs["train_self.logs"][self.logs["best_self.logs"]["step"]][
            "validate_every"] - 1
        best_loss = self.logs["best_self.logs"]["val_loss"]

        plt.plot((best_step, best_step), (0, best_loss), linestyle="--", color="y", label="Chosen Point")
        plt.plot((0, best_step), (best_loss, best_loss), linestyle="--", color="y")

        if self.config.rec_test:
            test_loss = np.array([self.logs["test_self.logs"][i]["test_loss"] for i in self.logs["test_self.logs"]])
            best_test_step = np.argmin(test_loss)
            best_test_loss = test_loss[best_test_step]
            plt.plot(steps, test_loss, label="Test")
            plt.plot((best_test_step, best_test_step), (0, best_test_loss), linestyle="--", color="r", label="Chosen Point")
            plt.plot((0, best_test_step), (best_test_loss, best_test_loss), linestyle="--", color="r")

        plt.xlabel('Epochs')
        plt.ylabel('Loss Values')
        plt.title("Loss")
        plt.ylim([1, 2.5])
        plt.legend()
        plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/loss.png")
        plt.show()

    def sleep_plot_k(self):

        train_k = np.array([self.logs["train_self.logs"][i]["train_k"] for i in self.logs["train_self.logs"]])
        val_k = np.array([self.logs["val_self.logs"][i]["val_k"] for i in self.logs["val_self.logs"]])
        steps = np.array([i / self.logs["train_self.logs"][i]["validate_every"] for i in self.logs["train_self.logs"]]) - 1

        plt.figure()
        plt.plot(steps, train_k, label="Train")
        plt.plot(steps, val_k, label="Valid")

        best_step = self.logs["best_self.logs"]["step"] / self.logs["train_self.logs"][self.logs["best_self.logs"]["step"]][
            "validate_every"] - 1
        best_k = self.logs["best_self.logs"]["val_k"]

        plt.plot((best_step, best_step), (0, best_k), linestyle="--", color="y", label="Chosen Point")
        plt.plot((0, best_step), (best_k, best_k), linestyle="--", color="y")

        if self.config.rec_test:
            test_k = np.array([self.logs["test_self.logs"][i]["test_k"] for i in self.logs["test_self.logs"]])
            best_test_step = np.argmax(test_k)
            best_test_k = test_k[best_test_step]
            plt.plot(steps, test_k, label="Test")
            plt.plot((best_test_step, best_test_step), (0, best_test_k), linestyle="--", color="r", label="Chosen Point")
            plt.plot((0, best_test_step), (best_test_k, best_test_k), linestyle="--", color="r")

        plt.xlabel('Epochs')
        plt.ylabel('Kappa')
        plt.title("Cohen's kappa")
        plt.legend()
        plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/kappa.png")
        plt.show()

    def sleep_plot_f1(self):
        train_f1 = np.array([self.logs["train_self.logs"][i]["train_f1"] for i in self.logs["train_self.logs"]])
        val_f1 = np.array([self.logs["val_self.logs"][i]["val_f1"] for i in self.logs["val_self.logs"]])
        steps = np.array([i / self.logs["train_self.logs"][i]["validate_every"] for i in self.logs["train_self.logs"]]) - 1

        plt.figure()
        plt.plot(steps, train_f1, label="Train")
        plt.plot(steps, val_f1, label="Valid")

        best_step = self.logs["best_self.logs"]["step"] / self.logs["train_self.logs"][self.logs["best_self.logs"]["step"]][
            "validate_every"] - 1
        best_f1 = self.logs["best_self.logs"]["val_f1"]

        plt.plot((best_step, best_step), (0, best_f1), linestyle="--", color="y", label="Chosen Point")
        plt.plot((0, best_step), (best_f1, best_f1), linestyle="--", color="y")

        if self.config.rec_test:
            test_f1 = np.array([self.logs["test_self.logs"][i]["test_f1"] for i in self.logs["test_self.logs"]])
            best_test_step = np.argmax(test_f1)
            best_test_f1 = test_f1[best_test_step]
            plt.plot(steps, test_f1, label="Test")
            plt.plot((best_test_step, best_test_step), (0, best_test_f1), linestyle="--", color="r", label="Chosen Point")
            plt.plot((0, best_test_step), (best_test_f1, best_test_f1), linestyle="--", color="r")

        plt.xlabel('Epochs')
        plt.ylabel('F1')
        plt.title("Training progress: F1 ")
        plt.legend()
        plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/f1.png")
        plt.show()

    def sleep_plot_eeg(self, data):
        time = np.arange(0, 30 - 1 / 900, 30 / 900)
        plt.figure("EEG Window")
        data = data.squeeze()
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(time, data[0][i])
        plt.show()

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

    def get_teacher_estimations(self):
        self.model.eval()
        valid_loss = 0
        tts, preds, ids = [], [], []
        # hidden = None
        with torch.no_grad():
            for batch_idx, (data, target, init, id) in enumerate(self.data_loader.total_loader):

                views = [data[i].float().to(self.device) for i in range(len(data))]
                label = target.to(self.device).flatten()
                pred = self.get_predictions_time_series(views, init)

                tts.append(label.cpu())
                preds.append(pred.cpu())
                ids.append(id.flatten().cpu())

        tts = torch.cat(tts,dim=0).numpy()
        preds = torch.cat(preds,dim=0).numpy()
        ids = torch.cat(ids,dim=0).numpy()

        print(min(ids))
        print(max(ids))
        print(ids.shape)
        print(preds.shape)
        print(self.data_loader.total_loader.dataset.cumulative_lengths)
        teacher_predictions = {}
        for id in range(len(ids)):
            for c in range(1,len(self.data_loader.total_loader.dataset.cumulative_lengths)):
                if ids[id] >= self.data_loader.total_loader.dataset.cumulative_lengths[c-1] and self.data_loader.total_loader.dataset.cumulative_lengths[c] > ids[id]:
                    if self.data_loader.total_loader.dataset.dataset[0][c-1] not in teacher_predictions:
                        teacher_predictions[self.data_loader.total_loader.dataset.dataset[0][c-1]] = {}
                    # print(teacher_predictions[self.data_loader.total_loader.dataset.dataset[0][c-1]])
                    # print(preds[id])
                    teacher_predictions[self.data_loader.total_loader.dataset.dataset[0][c-1]][id - self.data_loader.total_loader.dataset.cumulative_lengths[c-1]] = preds[id]
                    break

        import pickle


        with open('./teacher_predictions.pickle', 'wb') as handle:
            pickle.dump(teacher_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)


        print(len(teacher_predictions.keys()))
        print(tts.shape)
        print(preds.shape)
        print(ids.shape)

        print(tts[0:5])
        print(preds[0:5])
        print(ids[0:5])

    def check_energies_per_class(self):

        total = [[], [], [], [], []]
        with torch.no_grad():
            pbar = tqdm(enumerate(self.data_loader.valid_loader), desc="Validation", leave=False,
                        disable=True, position=1)
            for batch_idx, batch in pbar:
                data, target, init = batch[0], batch[1], batch[2]
                a = data[0].flatten(start_dim=0, end_dim=2 ).squeeze()
                target = target.flatten()
                for i, j in enumerate(target):
                    total[j.item()].append(a[i].unsqueeze(dim=0))

        x = np.arange(129)
        col  = ["black", "blue",  "orange", "yellow", "red"]
        from random import sample
        for spectrum_suspace in [[1,35],[35,80],[80,129]]:
            print("Spectrum is {}-{}".format(spectrum_suspace[0],spectrum_suspace[1]))
            total_sub = []
            for i in range(len(total)):
                total_sub.append(torch.cat(total[i], dim=0)[:,spectrum_suspace[0]:spectrum_suspace[1]])
                total_sub[i] = (total_sub[i]-total_sub[i].mean())/total_sub[i].std()
                # print(total[i].shape)

            diffs = torch.zeros([5,5])
            for i in range(len(total_sub)):
                for j in range(len(total_sub)):
                    hotmat = torch.einsum("ijd,mjd->im",total_sub[i],total_sub[j])
                    diffs[i,j] = hotmat.mean()
                    # print("Difference {}-{} is {}".format(i, j, diffs[i,j]))

            print(diffs)
            plt.imshow(diffs.numpy(), cmap='Blues', interpolation='none')
            plt.title("Spectrum is {}-{}".format(spectrum_suspace[0],spectrum_suspace[1]))
            plt.colorbar()
            plt.show()
                #
                #
                # t1 = np.array(total[i]).squeeze()
                # t2 = np.array(total[j]).squeeze()
                # num_cores = 8
                # t1_subsample = sample(list(np.arange(len(t1))),5)
                # t2_subsample = sample(list(np.arange(len(t2))),5)
                # diff_scramble = Parallel(n_jobs=num_cores)(delayed(self._parallel_diff_calc)(t1[el1], t2[t2_subsample,:]) for el1 in tqdm(t1_subsample, "Diff calc"))
                # actual_diff = self.gather_diffs(diff_scramble)
                # print("Difference {}-{} is {}".format(i, j, actual_diff))

    def _parallel_diff_calc(self, el1, t2):
        diff = 0
        sum = 0
        for el2 in t2:
            diff += np.linalg.norm(el1 - el2)
            sum += 1
        return {"diff":diff, "sum":sum}

    def gather_diffs(self, diffs):
        diff, sum = 0, 0
        for i in diffs:
            if isinstance(i,dict) and "diff" in i and "sum" in i:
                diff += i["diff"]
                sum += i["sum"]
        return diff/sum

        #
        #     print(t.shape)
        #     print(t.mean(axis=0))
        #     plt.plot(x,  t.mean(axis=0)[0], 'o', color=col[i])
        # plt.show()

