import torch
import time
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import numpy as np
from collections import defaultdict
from colorama import Fore, Back, Style
import wandb
import logging

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
                except:
                    raise Exception("Problem in model saving")

    def sleep_save(self, file_name: str="checkpoint.pth.tar", post_test_results: dict=False):
            """
            Checkpoint saver
            :param file_name: name of the checkpoint file
            :param post_test_results: dict containing the results of the test evaluation after training, default value False indicates that they are not provided
            :return:
            """
            save_dict = {}
            savior = {}
            savior["model_state_dict"] = self.agent.model.state_dict()
            savior["best_model_state_dict"] = self.agent.best_model.state_dict()
            savior["optimizer_state_dict"] = self.agent.optimizer.state_dict()
            savior["logs"] = self.agent.logs
            if post_test_results:
                savior["post_test_results"] = post_test_results
            savior["metrics"] = self.agent.data_loader.metrics
            savior["configs"] = self.agent.config

            save_dict.update(savior)

            try:
                pass
                # torch.save(save_dict, file_name)
            except:
                raise Exception("Problem in model saving")
            # self.save_encoder()

    def monitoring(self):

        train_metrics = self.agent.evaluators.train_evaluator.evaluate()
        val_metrics = self.agent.evaluators.val_evaluator.evaluate()

        self._find_learning_rate()

        self._update_train_val_logs(train_metrics = train_metrics, val_metrics = val_metrics)
        wandb.log({"train": train_metrics, "val": val_metrics})

        is_best = self.agent.evaluators.val_evaluator.is_best(metrics=val_metrics, best_logs=self.agent.logs["best_logs"])
        not_saved = True
        if is_best:
            self._update_best_logs(current_step = self.agent.logs["current_step"], val_metrics = val_metrics)
            self.agent.best_model.load_state_dict(self.agent.model.state_dict())
            if self.agent.config.training_params.rec_test:
                self._test_n_update()

            self.agent.logs["steps_no_improve"] = 0
            self.sleep_save()
            not_saved = False
        else:
            self.agent.logs["steps_no_improve"] += 1
            if self.agent.config.training_params.rec_test and self.agent.config.training_params.test_on_bottoms:
                self._test_n_update()

        self._early_stop_check_n_save(not_saved)

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

        self.print_valid_results(val_metrics, current_step)

    def print_valid_results(self, val_metrics, current_step):

        if self.agent.config.training_params.verbose:
            step = int(current_step / self.agent.config.early_stopping.validate_every)
            if not self.agent.config.training_params.tdqm_disable and not self.agent.trainer.end_of_epoch_check: print()

            message = Fore.WHITE + "Epoch {0:d} step {1:d} with ".format(self.agent.logs["current_epoch"], step)
            if "loss" in val_metrics:
                for i, v in val_metrics["loss"].items(): message += Fore.RED + "{} : {:.6f} ".format(i,v)
            if "acc" in val_metrics:
                for i, v in val_metrics["acc"].items(): message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i,v*100)
            if "f1" in val_metrics:
                for i, v in val_metrics["f1"].items(): message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i,v*100)
            if "k" in val_metrics:
                for i, v in val_metrics["k"].items(): message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i,v)
            # if "val_perclassf1" in val_metrics:
            #     for i, v in val_metrics["val_perclassf1"].items(): message += Fore.BLUE + "F1_perclass_{}: {} ".format(i,"{}".format(str(list((v*100).round(2)))))

            logging.info(message)

    def _test_n_update(self):
        test_metrics = self.agent.evaluators.test_evaluator.evaluate()

        self.agent.logs["test_logs"][self.agent.logs["current_step"]] = test_metrics
        self.print_valid_results(test_metrics, self.agent.logs["current_step"])

    def _early_stop_check_n_save(self, not_saved):

        training_cycle = (self.agent.logs["current_step"] // self.agent.config.early_stopping.validate_every)
        if not_saved and training_cycle % self.agent.config.early_stopping.save_every_valstep == 0:
            # Some epochs without improvement have passed, we save to avoid losing progress even if its not giving new best
            self.sleep_save()

        if training_cycle == self.agent.config.early_stopping.n_steps_stop_after:
            # After 'n_steps_stop_after' we need to start counting till we reach the earlystop_threshold
            self.steps_at_earlystop_threshold = self.agent.logs["steps_no_improve"] # we dont need to initialize that since training_cycle > self.agent.config.n_steps_stop_after will not be true before ==

        if training_cycle > self.agent.config.early_stopping.n_steps_stop_after and self.agent.logs["steps_no_improve"] >= self.agent.config.early_stopping.n_steps_stop:
            self.agent.evaluators.train_evaluator.set_early_stop()


