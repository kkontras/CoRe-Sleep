import torch
import torch.nn as nn

from graphs.models.attention_models.windowFeature_base import *
from colorama import Fore, Back, Style
import torch.optim as optim
from utils.schedulers.no_scheduler import No_Scheduler
from utils.schedulers.warmup_scheduler import WarmupScheduler
from utils.optimizers.lamp_optimizer import Lamb
import wandb
from graphs.models.xsleepnet.xsleepnet_pytorch.xsleepnet_pytorch import XSleepnet
from asxeta.sleepTranformer_Zoi import SleepTransformer

class Loader():

    def __init__(self, agent):
        self.agent = agent

    def load_pretrained_models(self):
        if "pretrained_model" in self.agent.config.model:
            if self.agent.config.model.pretrained_model["use"] and not self.agent.config.model.load_ongoing:
                print("Loading pretrained model from file {}".format(self.agent.config.model.pretrained_model["dir"]))
                checkpoint = torch.load(self.agent.config.model.pretrained_model["dir"])
                self.agent.model.load_state_dict(checkpoint["model_state_dict"])
                self.agent.best_model.load_state_dict(checkpoint["model_state_dict"])

    def load_models_n_optimizer(self):
        enc = self.sleep_load_encoder(enc_args=self.agent.config.model.encoders)
        model_class = globals()[self.agent.config.model.model_class]
        # self.agent.model = model_class(enc, args = self.agent.config.model.args)
        self.agent.model = nn.DataParallel(model_class(encs = enc, args = self.agent.config.model.args), device_ids=[torch.device(i) for i in self.agent.config.training_params.gpu_device])
        self.agent.best_model = copy.deepcopy(self.agent.model)
        # self.agent.best_model = nn.DataParallel(model_class(enc, args = self.agent.config.model.args), device_ids=[torch.device(i) for i in self.agent.config.gpu_device])
        self._my_numel(self.agent.model, verbose=True)
        if self.agent.config.optimizer.type == "Adam":
            list_of_params = [{'params':self.agent.model.module.fc_out.parameters()}]
            for i in range(len(self.agent.config.model.encoders)):

                enc_i = getattr(self.agent.model.module, "enc_{}".format(i))
                if "lr" in self.agent.config.model.encoders[i].args:
                    print("Encoder {} has diff learning {}".format(i, self.agent.config.model.encoders[i].args.lr))
                    list_of_params.append({'params':enc_i.parameters(), "lr": self.agent.config.model.encoders[i].args.lr})
                else:
                    list_of_params.append({'params':enc_i.parameters()})

            self.agent.optimizer = optim.Adam(list_of_params,
                                              lr=self.agent.config.optimizer.learning_rate,
                                              betas=(self.agent.config.optimizer.beta1, self.agent.config.optimizer.beta2),
                                              eps=1e-07,
                                              weight_decay=self.agent.config.optimizer.weight_decay)

        elif self.agent.config.optimizer.type == "MomentumSGD":
            self.agent.optimizer = optim.SGD(self.agent.model.parameters(),
                                    lr=self.agent.config.optimizer.learning_rate,
                                    momentum=self.agent.config.optimizer.momentum)
        elif self.agent.config.optimizer.type == "Adadelta":
            self.agent.optimizer = optim.Adadelta(self.agent.model.parameters(),
                                            lr=self.agent.config.optimizer.learning_rate,
                                            rho=0.9,
                                            eps=1e-06,
                                            weight_decay=self.agent.config.optimizer.weight_decay)
        elif self.agent.config.optimizer.type == "Lamb":
            self.agent.optimizer = Lamb(self.agent.model.parameters(),
                                  lr=self.agent.config.optimizer.learning_rate,
                                  betas=(self.agent.config.optimizer.beta1, self.agent.config.optimizer.beta2),
                                  eps = 1e-07,
                                  weight_decay=self.agent.config.optimizer.weight_decay)

        self.load_pretrained_models()

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

    def get_scheduler(self):
        if self.agent.config.scheduler.type == "cyclic":
            after_scheduler = optim.lr_scheduler.CyclicLR(self.agent.optimizer, base_lr=self.agent.config.optimizer.learning_rate, max_lr=self.agent.config.scheduler.max_lr, cycle_momentum=False)

            self.agent.scheduler = WarmupScheduler(optimizer=self.agent.optimizer,
                                                   base_lr=self.agent.config.optimizer.learning_rate,
                                                   n_warmup_steps=self.agent.config.scheduler.warm_up * self.agent.config.early_stopping.validate_every,
                                                   after_scheduler=after_scheduler)

        elif self.agent.config.scheduler.type == "cosanneal":

            after_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.agent.optimizer, T_0=4, T_mult=2)
            self.agent.scheduler = WarmupScheduler(optimizer=self.agent.optimizer,
                                                   base_lr=self.agent.config.optimizer.learning_rate,
                                                   n_warmup_steps=self.agent.config.scheduler.warm_up * self.agent.config.early_stopping.validate_every,
                                                   after_scheduler=after_scheduler)

        else:
            self.agent.scheduler = No_Scheduler(base_lr=self.agent.config.optimizer.learning_rate)

    def sleep_load(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        print("Loading from file {}".format(file_name))
        checkpoint = torch.load(file_name)
        self.agent.model.load_state_dict(checkpoint["model_state_dict"])
        self.agent.best_model.load_state_dict(checkpoint["best_model_state_dict"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.agent.logs = checkpoint["logs"]
        self.agent.data_loader.load_metrics_ongoing(checkpoint["metrics"])
        self.agent.data_loader.weights = self.agent.logs["weights"]
        self.agent.weights = self.agent.data_loader.weights

        for step in self.agent.logs["train_logs"]:
            wandb.log({"train": self.agent.logs["train_logs"][step], "val":  self.agent.logs["val_logs"][step]}, step=step)
            for i, lr in enumerate(self.agent.logs["train_logs"][step]["learning_rate"]):
                wandb.log({"lr": lr, "val":  self.agent.logs["val_logs"][step]}, step=i+ step - self.agent.config.early_stopping.validate_every)

            if "val_acc" in self.agent.logs["val_logs"][step]:

                for pred_key in list(self.agent.logs["train_logs"][step]["train_acc"].keys()):
                    wandb.log({"train.train_f1_perclass.{}.W".format(pred_key): self.agent.logs["train_logs"][step]["train_perclassf1"][pred_key][0],
                               "train.train_f1_perclass.{}.N1".format(pred_key): self.agent.logs["train_logs"][step]["train_perclassf1"][pred_key][1],
                               "train.train_f1_perclass.{}.N2".format(pred_key): self.agent.logs["train_logs"][step]["train_perclassf1"][pred_key][2],
                               "train.train_f1_perclass.{}.N3".format(pred_key): self.agent.logs["train_logs"][step]["train_perclassf1"][pred_key][3]})
                for pred_key in list(self.agent.logs["val_logs"][step]["val_acc"].keys()):
                    wandb.log({"val.val_f1_perclass.{}.W".format(pred_key): self.agent.logs["val_logs"][step]["val_perclassf1"][pred_key][0],
                               "val.val_f1_perclass.{}.N1".format(pred_key): self.agent.logs["val_logs"][step]["val_perclassf1"][pred_key][1],
                               "val.val_f1_perclass.{}.N2".format(pred_key): self.agent.logs["val_logs"][step]["val_perclassf1"][pred_key][2],
                               "val.val_f1_perclass.{}.N3".format(pred_key): self.agent.logs["val_logs"][step]["val_perclassf1"][pred_key][3],
                               "val.val_f1_perclass.{}.REM".format(pred_key): self.agent.logs["val_logs"][step]["val_perclassf1"][pred_key][4]})


        # if "loss_func" in checkpoint:
        #     self.loss = checkpoint["loss_func"]
        # else:
        self.agent.loss = nn.CrossEntropyLoss(self.agent.weights.to(self.agent.device))

        print("Model has loaded successfully")
        print("Metrics have been loaded")
        print("Loaded loss weights are:", self.agent.weights)

        message = Fore.WHITE + "The best in step: {} so far \n".format(
            int(self.agent.logs["best_logs"]["step"] / self.agent.config.early_stopping.validate_every))

        if "val_loss" in self.agent.logs["best_logs"]:
            for i, v in self.agent.logs["best_logs"]["val_loss"].items(): message += Fore.RED + "{} : {:.6f} ".format(i,v)
        if "val_acc" in self.agent.logs["best_logs"]:
            for i, v in self.agent.logs["best_logs"]["val_acc"].items(): message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i, v * 100)
        if "val_f1" in self.agent.logs["best_logs"]:
            for i, v in self.agent.logs["best_logs"]["val_f1"].items(): message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i, v * 100)
        if "val_k" in self.agent.logs["best_logs"]:
            for i, v in self.agent.logs["best_logs"]["val_k"].items(): message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i, v)
        if "val_perclassf1" in self.agent.logs["best_logs"]:
            for i, v in self.agent.logs["best_logs"]["val_perclassf1"].items(): message += Fore.BLUE + "F1_perclass_{}: {} ".format(i, "{}".format( str(list((v * 100).round(2)))))

        #     print(message)
        # for i, v in self.agent.logs["best_logs"]["val_loss"].items(): message += Fore.RED + "{} : {:.6f} ".format(i, v)
        # for i, v in self.agent.logs["best_logs"]["val_acc"].items(): message += Fore.LIGHTBLUE_EX +"Acc_{}: {:.2f} ".format(i, v * 100)
        # for i, v in self.agent.logs["best_logs"]["val_f1"].items(): message += Fore.LIGHTGREEN_EX +"F1_{}: {:.2f} ".format(i, v * 100)
        # for i, v in self.agent.logs["best_logs"]["val_k"].items(): message += Fore.LIGHTGREEN_EX +"K_{}: {:.4f} ".format(i, v)
        # for i, v in self.agent.logs["best_logs"]["val_perclassf1"].items(): message += Fore.BLUE +"F1_perclass_{}: {} ".format(i, "{}".format(str(list((v * 100).round(2)))))

        print(message)

    def sleep_load_encoder(self, enc_args):
        encs = []
        for num_enc in range(len(enc_args)):
            enc_class = globals()[enc_args[num_enc]["model"]]
            args = enc_args[num_enc]["args"]
            print(enc_class)
            if "encoders" in enc_args[num_enc]:
                enc_enc = self.sleep_load_encoder(enc_args[num_enc]["encoders"])
                enc = enc_class(encs = enc_enc, args = args)
            else:
                enc = enc_class(args = args)
            enc = nn.DataParallel(enc, device_ids=[torch.device(i) for i in self.agent.config.training_params.gpu_device])

            if enc_args[num_enc]["pretrainedEncoder"]["use"]:
                print("Loading encoder from {}".format(enc_args[num_enc]["pretrainedEncoder"]["dir"]))
                checkpoint = torch.load(enc_args[num_enc]["pretrainedEncoder"]["dir"])
                if "encoder_state_dict" in checkpoint:
                    enc.load_state_dict(checkpoint["encoder_state_dict"])
                elif "model_state_dict" in checkpoint:
                    enc.load_state_dict(checkpoint["model_state_dict"])

            encs.append(enc)
        return encs