# import sys
# sys.path.append("/users/sista/kkontras/Documents/Sleep_Project/")
from utils.config import process_config
import torch
import torch.nn as nn
import copy
from datasets.sleepset import *
from graphs.models.attention_models.windowFeature_base import *
from graphs.models.custom_layers.eeg_encoders import *
from graphs.models.attention_models.BLIP import *
from colorama import init, Fore, Back, Style

class Importer():
    def __init__(self, config_name:str, device:str="cuda:0", fold:int=0):
        print("Loading config from {}".format(config_name))
        self.config = process_config(config_name, False)
        self.device = device
        self.fold = fold

    def load_checkpoint(self):
        self.config.model.save_dir = self.config.model.save_dir.format(self.fold)
        if "model" not in self.config:
            print("Loading from {}".format(self.config.save_dir))
            self.checkpoint = torch.load(self.config.save_dir, map_location="cpu")
        else:
            print("Loading from {}".format(self.config.model.save_dir))
            self.checkpoint = torch.load(self.config.model.save_dir, map_location="cpu")

    def change_config(self, attr, value, c = None):
        if c == None: c = self.config
        attr_splits = attr.split(".")
        for attr_split in attr_splits[:-1]:
            c = getattr(c, attr_split)
        setattr(c, attr_splits[-1], value)

    def get_dataloaders(self):

        dataloader = globals()[self.config.dataset.dataloader_class]
        data_loader = dataloader(config=self.config)
        data_loader.load_metrics_ongoing(self.checkpoint["metrics"])
        data_loader.weights = self.checkpoint['logs']["weights"]

        return data_loader

    def get_model(self, model = None, return_model:str = "best_model"):

        if not model:
            model_class = globals()[self.config.model.model_class]

            enc = self._sleep_load_encoder(encoders=self.config.model.encoders)
            model = model_class(enc, args=self.config.model.args)

            model = model.to(self.device)
            model = nn.DataParallel(model, device_ids=[torch.device(i) for i in self.config.training_params.gpu_device])

        if return_model == "untrained_model":
            return model
        elif return_model == "best_model":
            best_model = copy.deepcopy(model)
            best_model.load_state_dict(self.checkpoint["best_model_state_dict"])
            return best_model
        elif return_model == "running_model":
            running_model = copy.deepcopy(model)
            running_model.load_state_dict(self.checkpoint["model_state_dict"])
            return running_model
        else:
            raise ValueError(
                'Return such model does not exits as option, choose from "best_model","running_model", "untrained_model" ')

    def print_progress(self, multi_fold_results, print_entropy=False, latex_version=False):
        # init(convert=True)

        val_metrics = self.checkpoint["logs"]["best_logs"]
        #This is meant for old experiments
        # if type(val_metrics["val_loss"]) !=dict:
        #     return self.print_progress_old(multi_fold_results)

        print("-- Best Validation --")
        latex_message = {}
        if "val_acc" not in val_metrics:
            step = int(val_metrics["step"] / self.config.early_stopping.validate_every)
            message = Style.BRIGHT + Fore.WHITE + "Step: {}, No_improve: {} ".format(step, self.checkpoint["logs"][
                "steps_no_improve"])
            if "val_loss" in val_metrics:
                for i, v in val_metrics["val_loss"].items():
                    message += Fore.RED + "{} : {:.6f} ".format(i, val_metrics["val_loss"][i])
            print(message + Style.RESET_ALL)

        else:
            for pred in val_metrics["val_acc"]:
                step = int(val_metrics["step"] / self.config.early_stopping.validate_every)
                message = Style.BRIGHT + Fore.WHITE + "Step: {}, No_improve: {} ".format( step, self.checkpoint["logs"]["steps_no_improve"])
                latex_message[pred] = "{} & ".format(pred)
                if "val_loss" in val_metrics:
                    for i, v in val_metrics["val_loss"].items():
                        if pred in i or i =="total":
                            message += Fore.RED + "{} : {:.6f} ".format(i, val_metrics["val_loss"][i])
                # if "val_loss" in val_metrics:
                #     for i, v in val_metrics["val_loss"].items(): message += Fore.RED + "{} : {:.6f} ".format(i, v)
                if "val_acc" in val_metrics:
                    if pred in val_metrics["val_acc"]:
                        message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(pred, val_metrics["val_acc"][pred] * 100)
                        latex_message[pred] += " {:.1f} &".format(val_metrics["val_acc"][pred] * 100)
                if "val_k" in val_metrics:
                    if pred in val_metrics["val_k"]:
                        message += Fore.LIGHTGREEN_EX + "K_{}: {:.3f} ".format(pred, val_metrics["val_k"][pred])
                        latex_message[pred] += " {:.3f} &".format(val_metrics["val_k"][pred])
                if "val_f1" in val_metrics:
                    if pred in val_metrics["val_f1"]:
                        message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(pred, val_metrics["val_f1"][pred] * 100)
                        latex_message[pred] += " {:.1f} &".format(val_metrics["val_f1"][pred] * 100)
                if "val_perclassf1" in val_metrics:
                    if pred in val_metrics["val_perclassf1"]:
                        message += Fore.BLUE + "F1_perclass_{}: {} ".format(pred,"{}".format(str(list((val_metrics["val_perclassf1"][pred] * 100).round(2)))))
                        for i in list((val_metrics["val_perclassf1"][pred] * 100).round(2)):
                            latex_message[pred] += " {:.1f} &".format(i)

                print(message+ Style.RESET_ALL)
                # print(latex_message[pred]+ Style.RESET_ALL)

        if self.config.training_params.rec_test:
            print("-- Best Test --")
            test_best_logs = self.checkpoint["logs"]["test_logs"][self.checkpoint["logs"]["best_logs"]["step"]]
            print("Acc: {0:.1f}, Kappa: {1:.3f}, F1: {2:.1f}, f1_per_class: {3:.1f} {4:.1f} {5:.1f} {6:.1f} {7:.1f}".format(
                test_best_logs["accuracy"]*100,
                test_best_logs["k"],
                test_best_logs["f1"]*100,
                test_best_logs["preclass_f1"][0]*100,
                test_best_logs["preclass_f1"][1]*100,
                test_best_logs["preclass_f1"][2]*100,
                test_best_logs["preclass_f1"][3]*100,
                test_best_logs["preclass_f1"][4]*100
            ))

        def _print_test_results(metrics, description, multi_fold_results, print_entropy=False):
            # description = "--- Post Test ---"
            latex_message = {}
            # message = Style.BRIGHT + Fore.WHITE + "{} ".format(description)
            print( Style.BRIGHT + Fore.WHITE +  "{} ".format(description))
            for pred in metrics["acc"]:
                message = "{} ".format(pred)
                latex_message[pred] = "{} & ".format(pred)

                if "acc" in metrics:
                    if pred in metrics["acc"]:
                        message += Fore.LIGHTBLUE_EX + "Acc: {:.1f} ".format(metrics["acc"][pred] * 100)
                        latex_message[pred] += " {:.1f} &".format(metrics["acc"][pred] * 100)

                if "k" in metrics:
                    if pred in metrics["k"]:
                        message += Fore.LIGHTGREEN_EX + "K: {:.3f} ".format(metrics["k"][pred])
                        latex_message[pred] += " {:.3f} &".format(metrics["k"][pred])

                if "f1" in metrics:
                    if pred in metrics["f1"]:
                        message += Fore.LIGHTGREEN_EX + "F1: {:.1f} ".format(metrics["f1"][pred] * 100)
                        latex_message[pred] += " {:.1f} &".format(metrics["f1"][pred] * 100)

                if "f1_perclass" in metrics:
                    if pred in metrics["f1_perclass"]:
                        message += Fore.BLUE + "F1_perclass: {} ".format("{}".format(
                            str(list((metrics["f1_perclass"][pred] * 100).round(1)))))
                        for i in list((metrics["f1_perclass"][pred] * 100).round(2)):
                            latex_message[pred] += " {:.1f} &".format(i)
                print(message + Style.RESET_ALL)
                print(latex_message[pred] + Style.RESET_ALL)

                #TODO: Make sure that this works to accumulate both the skipped and the normal cases, combined tags could get confused together
                multi_fold_results.update({self.fold: metrics})

                if print_entropy:
                    for pred in metrics["entropy"]:
                        message = ""
                        if "entropy" in metrics:
                            if pred in metrics["entropy"]:
                                message += Fore.LIGHTRED_EX + "E_{}: {:.4f} ".format(pred, metrics["entropy"][pred])
                        if "entropy_var" in metrics:
                            if pred in metrics["entropy_var"]:
                                message += Fore.LIGHTRED_EX + "E_var_{}: {:.4f} ".format(pred, metrics["entropy_var"][pred])
                        if "entropy_correct" in metrics:
                            if pred in metrics["entropy_correct"]:
                                message += Fore.LIGHTMAGENTA_EX + "EC_{}: {:.4f} ".format(pred, metrics["entropy_correct"][pred])
                        if "entropy_correct_var" in metrics:
                            if pred in metrics["entropy_correct_var"]:
                                message += Fore.LIGHTMAGENTA_EX + "EC_{}: {:.4f} ".format(pred, metrics["entropy_correct_var"][pred])
                        if "entropy_wrong" in metrics:
                            if pred in metrics["entropy_wrong"]:
                                message += Fore.LIGHTYELLOW_EX + "EW_{}: {:.4f} ".format(pred, metrics["entropy_wrong"][pred])
                        if "entropy_wrong_var" in metrics:
                            if pred in metrics["entropy_wrong_var"]:
                                message += Fore.LIGHTYELLOW_EX + "EW_var_{}: {:.4f} ".format(pred, metrics["entropy_wrong_var"][pred])

                        print(message + Style.RESET_ALL)
            return multi_fold_results


        if "post_test_results" in self.checkpoint:
            metrics = self.checkpoint["post_test_results"]
            multi_fold_results = _print_test_results(metrics=metrics, description="--- Post Test ---", print_entropy=print_entropy, multi_fold_results = multi_fold_results)
        # if "post_test_results_skipped" in self.checkpoint:
        #     metrics = self.checkpoint["post_test_results_skipped"]
        #     multi_fold_results = _print_test_results(metrics=metrics, description="--- Post Test Skipped ---", print_entropy=print_entropy, multi_fold_results= multi_fold_results)


        return multi_fold_results

    def print_progress_aggregated(self, multi_fold_results, latex_version=False):
        init_results = {"acc":[], "f1":[], "k":[], "f1_perclass":[] }
        aggregated_results = {}
        if multi_fold_results is None:
            return
        for fold_i in multi_fold_results:
            metrics = multi_fold_results[fold_i]
            for pred in metrics["acc"]:
                if pred not in aggregated_results:
                    aggregated_results[pred] = copy.deepcopy(init_results)
                aggregated_results[pred]["acc"].append(metrics["acc"][pred])
                aggregated_results[pred]["f1"].append(metrics["f1"][pred])
                aggregated_results[pred]["k"].append(metrics["k"][pred])
                aggregated_results[pred]["f1_perclass"].append(metrics["f1_perclass"][pred])

        for pred in aggregated_results:
            for k in aggregated_results[pred]:
                aggregated_results[pred][k] = {"mean": np.array(aggregated_results[pred][k]).mean(axis=0), "std": np.array(aggregated_results[pred][k]).std(axis=0)}
                # aggregated_results[pred][k] /= len(multi_fold_results)
            # # message = Style.BRIGHT + Fore.WHITE + "{} ".format(description)

        print("-- Aggregared Results {} folds --".format(len(multi_fold_results)))
        latex_message = {}
        latex_message_variance = {}
        for pred in aggregated_results:
            latex_message[pred] = "{} & ".format(pred)
            latex_message_variance[pred] = "{} & ".format(pred)
            message = Style.BRIGHT + Fore.WHITE + "AGG Results "

            message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(pred, aggregated_results[pred]["acc"]["mean"] * 100)
            latex_message[pred] += " {:.1f} {{\\tiny$\pm${:.1f}}} &".format(aggregated_results[pred]["acc"]["mean"] * 100, aggregated_results[pred]["acc"]["std"] * 100)

            message += Fore.LIGHTGREEN_EX + "K_{}: {:.3f} ".format(pred, aggregated_results[pred]["k"]["mean"])
            latex_message[pred] += " {:.3f} {{\\tiny$\pm${:.3f}}} &".format(aggregated_results[pred]["k"]["mean"], aggregated_results[pred]["k"]["std"])

            message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(pred, aggregated_results[pred]["f1"]["mean"] * 100)
            latex_message[pred] += " {:.1f} {{\\tiny$\pm${:.1f}}} &".format(aggregated_results[pred]["f1"]["mean"] * 100, aggregated_results[pred]["f1"]["std"] * 100)

            message += Fore.BLUE + "F1_perclass_{}: {} ".format(pred, "{}".format(str(list((aggregated_results[pred]["f1_perclass"]["mean"] * 100).round(2)))))
            for i in range(len(list((aggregated_results[pred]["f1_perclass"]["mean"])))):
                latex_message[pred] += " {:.1f} {{\\tiny$\pm${:.1f}}} &".format((aggregated_results[pred]["f1_perclass"]["mean"][i]*100).round(2),(aggregated_results[pred]["f1_perclass"]["std"][i]*100).round(2))
            # for i in list((aggregated_results[pred]["f1_perclass"]["std"] * 100).round(2)):
            #     latex_message_variance[pred] += " {:.1f} &".format(i)

            print(message + Style.RESET_ALL)
            if latex_version:
                print(latex_message[pred] + Style.RESET_ALL)
            # print(latex_message_variance[pred] + Style.RESET_ALL)

    def _my_numel(self, m: torch.nn.Module, only_trainable: bool = False, verbose = True):
        """
        returns the total number of parameters used by `m` (only counting
        shared parameters once); if `only_trainable` is True, then only
        includes parameters with `requires_grad = True`
        """
        parameters = list(m.parameters())
        if only_trainable:
            parameters = [p for p in parameters if p.requires_grad]
        unique = {p.data_ptr(): p for p in parameters}.values()
        model_total_params =  sum(p.numel() for p in unique)
        if verbose:
            print("Total number of trainable parameters are: {}".format(model_total_params))
        # for n, p in m.named_parameters()::
        #     if p.requires_grad:
        #         print(n, end=" - ")
        #         unique = {i.data_ptr(): i for i in p}.values()
        #         model_total_params = sum(i.numel() for i in unique)
        #         print(model_total_params)

        return model_total_params


    def print_progress_old(self, multi_fold_results):

        val_metrics = self.checkpoint["logs"]["best_logs"]

        print("-- Best Validation --")

        step = int(val_metrics["step"] / self.config.early_stopping.validate_every)
        message = Style.BRIGHT + Fore.WHITE + "Step: {}, No_improve: {} ".format( step, self.checkpoint["logs"]["steps_no_improve"])
        message += Fore.RED + "Loss : {:.6f} ".format(val_metrics["val_loss"])
        message += Fore.LIGHTBLUE_EX + "Acc: {:.2f} ".format(val_metrics["val_acc"] * 100)
        message += Fore.LIGHTGREEN_EX + "F1: {:.2f} ".format(val_metrics["val_f1"] * 100)
        message += Fore.LIGHTGREEN_EX + "K: {:.4f} ".format(val_metrics["val_k"])
        message += Fore.BLUE + "F1_perclass: {} ".format("{}".format(str(list((val_metrics["val_perclassf1"] * 100).round(2)))))
        print(message+ Style.RESET_ALL)


        if self.config.training_params.rec_test:
            print("-- Best Test --")
            test_best_logs = self.checkpoint["logs"]["test_logs"][self.checkpoint["logs"]["best_logs"]["step"]]
            print("Acc: {0:.1f}, Kappa: {1:.3f}, F1: {2:.1f}, f1_per_class: {3:.1f} {4:.1f} {5:.1f} {6:.1f} {7:.1f}".format(
                test_best_logs["accuracy"]*100,
                test_best_logs["k"],
                test_best_logs["f1"]*100,
                test_best_logs["preclass_f1"][0]*100,
                test_best_logs["preclass_f1"][1]*100,
                test_best_logs["preclass_f1"][2]*100,
                test_best_logs["preclass_f1"][3]*100,
                test_best_logs["preclass_f1"][4]*100
            ))

        if "post_test_results" in self.checkpoint:
            multi_fold_results.update({self.fold:self.checkpoint["post_test_results"]})
            print("-- Best Test --")
            print("Acc: {0:.1f}, Kappa: {1:.3f}, F1: {2:.1f}, f1_per_class: {3:.1f} {4:.1f} {5:.1f} {6:.1f} {7:.1f}".format(
                self.checkpoint["post_test_results"]["accuracy"]*100,
                self.checkpoint["post_test_results"]["k"],
                self.checkpoint["post_test_results"]["f1"]*100,
                self.checkpoint["post_test_results"]["preclass_f1"][0]*100,
                self.checkpoint["post_test_results"]["preclass_f1"][1]*100,
                self.checkpoint["post_test_results"]["preclass_f1"][2]*100,
                self.checkpoint["post_test_results"]["preclass_f1"][3]*100,
                self.checkpoint["post_test_results"]["preclass_f1"][4]*100
            ))

        return multi_fold_results

    def _sleep_load_encoder(self, encoders):
        # encs = []
        # for num_enc in range(len(encoders)):
        #
        #     enc_class = globals()[encoders[num_enc]["model"]]
        #     args = encoders[num_enc]["args"]
        #     enc = enc_class(args = args)
        #     enc = nn.DataParallel(enc, device_ids=[torch.device(0)])
        #
        #     if encoders[num_enc]["pretrainedEncoder"]["use"]:
        #         print("Loading encoder from {}".format(encoders[num_enc]["pretrainedEncoder"]["dir"]))
        #         checkpoint = torch.load(encoders[num_enc]["pretrainedEncoder"]["dir"])
        #         enc.load_state_dict(checkpoint["encoder_state_dict"])
        #     encs.append(enc)
        # return encs

        encs = []
        for num_enc in range(len(encoders)):
            enc_class = globals()[encoders[num_enc]["model"]]
            args = encoders[num_enc]["args"]
            print(enc_class)
            if "encoders" in encoders[num_enc]:
                enc_enc = self._sleep_load_encoder(encoders = encoders[num_enc]["encoders"])
                enc = enc_class(encs=enc_enc, args=args)
            else:
                enc = enc_class(args=args)
            enc = nn.DataParallel(enc, device_ids=[torch.device(i) for i in self.config.training_params.gpu_device])

            if encoders[num_enc]["pretrainedEncoder"]["use"]:
                print("Loading encoder from {}".format(encoders[num_enc]["pretrainedEncoder"]["dir"]))
                checkpoint = torch.load(encoders[num_enc]["pretrainedEncoder"]["dir"])
                if "encoder_state_dict" in checkpoint:
                    enc.load_state_dict(checkpoint["encoder_state_dict"])
                elif "model_state_dict" in checkpoint:
                    enc.load_state_dict(checkpoint["model_state_dict"])

            encs.append(enc)
        return encs
