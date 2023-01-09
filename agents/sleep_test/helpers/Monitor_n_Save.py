import torch
import time
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import numpy as np
from collections import defaultdict
from colorama import Fore, Back, Style
import wandb

class Monitor_n_Save():

    def __init__(self, agent):
        self.agent = agent

    def save_encoder(self):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        for enc_num in range(len(self.agent.config.model.encoders)):
            if "savetrainedEncoder" in self.agent.config.model.encoders[enc_num] and self.agent.config.model.encoders[enc_num]["savetrainedEncoder"]["save"]:
                save_dict = {}
                savior = {}
                if hasattr(self.agent.best_model.module, "encoder"):
                    savior["encoder_state_dict"] = self.agent.best_model.module.encoder.state_dict()
                elif hasattr(self.agent.best_model.module, "enc_{}".format(enc_num)):
                    enc = getattr(self.agent.best_model.module, "enc_{}".format(enc_num))
                    savior["encoder_state_dict"] = enc.state_dict()
                save_dict.update(savior)
                try:
                    torch.save(save_dict, self.agent.config.model.encoders[enc_num]["savetrainedEncoder"]["dir"])
                    print("Encoder saved successfully")
                except:
                    raise Exception("Problem in model saving")

    def sleep_save(self, file_name="checkpoint.pth.tar", verbose=False):
            """
            Checkpoint saver
            :param file_name: name of the checkpoint file
            :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
            :return:
            """
            save_dict = {}
            savior = {}
            savior["model_state_dict"] = self.agent.model.state_dict()
            savior["best_model_state_dict"] = self.agent.best_model.state_dict()
            savior["optimizer_state_dict"] = self.agent.optimizer.state_dict()
            savior["logs"] = self.agent.logs
            savior["metrics"] = self.agent.data_loader.metrics
            savior["configs"] = self.agent.config

            save_dict.update(savior)

            if self.agent.config.dataset.data_split.split_method == "patients_folds": file_name = file_name.format(self.agent.config.dataset.data_split.fold)

            try:
                torch.save(save_dict, file_name)
                second_filename = file_name.split(".")[-3] +"_cp.pth.tar"
                torch.save(save_dict, second_filename)
                # if verbose:
                #     print(Fore.WHITE + "Models has saved successfully in {}".format(file_name))
            except:
                raise Exception("Problem in model saving")

    def monitoring(self, train_metrics, val_metrics):

        self._find_learning_rate()

        self._update_train_val_logs(train_metrics = train_metrics, val_metrics = val_metrics)
        wandb.log({"train": train_metrics, "val": val_metrics})
        # if "train_acc" in train_metrics:
        #     for pred_key in list(train_metrics["train_acc"].keys()):
        #         print(train_metrics["train_perclassf1"][pred_key])
        #         wandb.log({"train.train_perclassf1.{}.W".format(pred_key): train_metrics["train_perclassf1"][pred_key][0],
        #                    "train.train_perclassf1.{}.N1".format(pred_key): train_metrics["train_perclassf1"][pred_key][1],
        #                    "train.train_perclassf1.{}.N2".format(pred_key): train_metrics["train_perclassf1"][pred_key][2],
        #                    "train.train_perclassf1.{}.N3".format(pred_key): train_metrics["train_perclassf1"][pred_key][3],
        #                    "train.train_perclassf1.{}.REM".format(pred_key): train_metrics["train_perclassf1"][pred_key][4]})
        # if "val_acc" in val_metrics:
        #     for pred_key in list(val_metrics["val_acc"].keys()):
        #         wandb.log({"val.val_perclassf1.{}.W".format(pred_key): val_metrics["val_perclassf1"][pred_key][0],
        #                    "val.val_perclassf1.{}.N1".format(pred_key): val_metrics["val_perclassf1"][pred_key][1],
        #                    "val.val_perclassf1.{}.N2".format(pred_key): val_metrics["val_perclassf1"][pred_key][2],
        #                    "val.val_perclassf1.{}.N3".format(pred_key): val_metrics["val_perclassf1"][pred_key][3],
        #                    "val.val_perclassf1.{}.REM".format(pred_key): val_metrics["val_perclassf1"][pred_key][4]})

        #Flag if its saved dont save it again on $save_every
        not_saved = True

        #If we have a better validation loss
        if (val_metrics["val_loss"]["total"] < self.agent.logs["best_logs"]["val_loss"]["total"]):
            self._update_best_logs(current_step = self.agent.logs["current_step"], val_metrics = val_metrics)
            self.agent.best_model.load_state_dict(self.agent.model.state_dict())
            if self.agent.config.training_params.rec_test:
                self._test_n_update()

            self.agent.logs["saved_step"] = self.agent.logs["current_step"]
            self.agent.logs["steps_no_improve"] = 0
            self.sleep_save(self.agent.config.model.save_dir, verbose = True)
            self.save_encoder()
            not_saved = False
        else:
            self.agent.logs["steps_no_improve"] += 1
            if self.agent.config.training_params.rec_test and self.agent.config.training_params.test_on_tops:
                self._test_n_update()

        return self._early_stop_check_n_save(not_saved)

    def checkpointing(self, batch_loss, predictions, targets, incomplete_idxs=None):

        if "softlabels" in self.agent.config.dataset and self.agent.config.dataset.softlabels:
            targets_tens = torch.cat(targets).argmax(dim=1).cpu().numpy().flatten()
        else:
            targets_tens = torch.cat(targets).cpu().numpy().flatten()

        # TODO: Check if None values in normal training have any problem here.
        if None not in incomplete_idxs:
            incomplete_idxs_cat = {pred_key: torch.cat([incomplete_idxs[i][pred_key] for i in range(len(incomplete_idxs))]).cpu().numpy().flatten() for pred_key in incomplete_idxs[0]}
            target_dict = { pred_key: targets_tens[incomplete_idxs_cat[pred_key]] for pred_key in incomplete_idxs_cat}
        else:
            target_dict = { pred_key: targets_tens for pred_key in predictions[0]}

        total_preds, train_metrics  = {}, defaultdict(dict)
        train_metrics["train_loss"] = dict(batch_loss)
        for pred_key in predictions[0]:
            total_preds[pred_key] = np.concatenate([pred[pred_key] for pred in predictions if pred_key in pred],axis=0).argmax(axis=-1)
            train_metrics["train_acc"][pred_key] =  np.equal(target_dict[pred_key], total_preds[pred_key]).sum() / len(target_dict[pred_key])
            train_metrics["train_f1"][pred_key] = f1_score(total_preds[pred_key], target_dict[pred_key], average="macro")
            train_metrics["train_k"][pred_key] = cohen_kappa_score(total_preds[pred_key], target_dict[pred_key])
            train_metrics["train_perclassf1"][pred_key] = f1_score(total_preds[pred_key], target_dict[pred_key], average=None)

        train_metrics = dict(train_metrics) #Avoid passing empty dicts to logs, better return an error!

        val_metrics = self.agent.validator_tester.sleep_validate()
        early_stop = self.monitoring(train_metrics=train_metrics, val_metrics=val_metrics)
        return early_stop, val_metrics["val_loss"]

    def _find_learning_rate(self):
        for param_group in self.agent.optimizer.param_groups:
            self.lr = param_group['lr']

    def _update_train_val_logs(self, train_metrics, val_metrics):

        train_metrics.update({  "validate_every": self.agent.config.early_stopping.validate_every,
                                "batch_size": self.agent.config.training_params.batch_size,
                                "learning_rate": self.agent.scheduler.lr_history[
                                                  max(self.agent.logs["current_step"] - self.agent.config.early_stopping.validate_every, 0):
                                                  self.agent.logs["current_step"]]})

        self.agent.logs["val_logs"][self.agent.logs["current_step"]] = val_metrics
        self.agent.logs["train_logs"][self.agent.logs["current_step"]] = train_metrics

    def _update_best_logs(self, current_step, val_metrics):

        val_metrics.update({"step": current_step})
        self.agent.logs["best_logs"] = val_metrics

        if self.agent.config.training_params.verbose:
            step = int(current_step / self.agent.config.early_stopping.validate_every)
            if not self.agent.config.training_params.tdqm_disable: print()

            message = Fore.WHITE + "Epoch {0:d} step {1:d} with ".format(self.agent.logs["current_epoch"], step)
            if "val_loss" in val_metrics:
                for i, v in val_metrics["val_loss"].items(): message += Fore.RED + "{} : {:.6f} ".format(i,v)
            if "val_acc" in val_metrics:
                for i, v in val_metrics["val_acc"].items(): message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i,v*100)
            if "val_f1" in val_metrics:
                for i, v in val_metrics["val_f1"].items(): message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i,v*100)
            if "val_k" in val_metrics:
                for i, v in val_metrics["val_k"].items(): message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i,v)
            if "val_perclassf1" in val_metrics:
                for i, v in val_metrics["val_perclassf1"].items(): message += Fore.BLUE + "F1_perclass_{}: {} ".format(i,"{}".format(str(list((v*100).round(2)))))
            print(message)

    def _print_epoch_metrics(self):
        if self.agent.config.training_params.verbose:
            print("Epoch {0:d}, N: {1:d}, lr: {2:.8f} Validation loss: {3:.6f}, accuracy: {4:.2f}% f1 :{5:.4f},  :{6:.4f}  Training loss: {7:.6f}, accuracy: {8:.2f}% f1 :{9:.4f}, k :{10:.4f},".format(
                self.agent.logs["current_epoch"],
                self.agent.logs["current_step"] * self.agent.config.training_params.batch_size * self.agent.config.dataset.seq_legth[0],
                self.lr,
                self.agent.logs["val_logs"][self.agent.logs["current_step"]]["val_loss"],
                self.agent.logs["val_logs"][self.agent.logs["current_step"]]["val_acc"] * 100,
                self.agent.logs["val_logs"][self.agent.logs["current_step"]]["val_f1"],
                self.agent.logs["val_logs"][self.agent.logs["current_step"]]["val_k"],
                self.agent.logs["train_logs"][self.agent.logs["current_step"]]["train_loss"],
                self.agent.logs["train_logs"][self.agent.logs["current_step"]]["train_acc"] * 100,
                self.agent.logs["train_logs"][self.agent.logs["current_step"]]["train_f1"],
                self.agent.logs["train_logs"][self.agent.logs["current_step"]]["train_f1"]))
    def _test_n_update(self):
        test_loss, test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens = self.agent.validator_tester.sleep_test()

        self.agent.logs["test_logs"][self.agent.logs["current_step"]] = {"test_loss": test_loss, "test_k": test_k,
                                                                         "test_f1": test_f1, "test_acc": test_acc,
                                                                         "test_spec": test_spec, "test_sens": test_conf,
                                                                         "test_conf": test_conf, "test_auc": test_auc,
                                                                         "test_perclass_f1": list(test_perclass_f1)}
        if self.agent.config.training_params.verbose:
            message = "Test"
            for i, v in enumerate(test_loss): message += "loss_{}: {:.6f} ".format(i,v)
            message += "accuracy: {0:.2f}% f1 :{1:.4f}, k :{2:.4f}, sens:{3:.4f}, spec:{4:.4f}, f1_per_class :{5:40}".format(test_acc * 100,
                    test_f1,
                    test_k, test_spec, test_sens,
                    "{}".format(str(list((test_perclass_f1*100).round(2)))))
            print(message)
    def _early_stop_check_n_save(self, not_saved):

        training_cycle = (self.agent.logs["current_step"] // self.agent.config.early_stopping.validate_every)
        if not_saved and training_cycle % self.agent.config.early_stopping.save_every == 0:
            # Some epochs without improvement have passed, we save to avoid losing progress even if its not giving new best
            self.sleep_save(self.agent.config.model.save_dir)
            self.agent.logs["saved_step"] = self.agent.logs["current_step"]
            self.save_encoder()

        if training_cycle == self.agent.config.early_stopping.n_steps_stop_after:
            # After 'n_steps_stop_after' we need to start counting till we reach the earlystop_threshold
            self.steps_at_earlystop_threshold = self.agent.logs["steps_no_improve"] # we dont need to initialize that since training_cycle > self.agent.config.n_steps_stop_after will not be true before ==

        early_stop = False
        # if "steps_at_earlystop_threshold" not in self.keys(): self.steps_at_earlystop_threshold = 150 #TODO: save this in logs or eliminate it
        if training_cycle > self.agent.config.early_stopping.n_steps_stop_after and self.agent.logs["steps_no_improve"] >= self.agent.config.early_stopping.n_steps_stop:
            early_stop = True
        return early_stop
