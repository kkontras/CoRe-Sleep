
import torch
from torch import nn
from torch.backends import cudnn

from utils.misc import print_cuda_statistics
from datasets.sleepset import *
from utils.deterministic_pytorch import deterministic
from agents.helpers.Loader import Loader
from agents.helpers.Monitor_n_Save import Monitor_n_Save
from agents.helpers.Trainer import Trainer
from agents.helpers.Validator_Tester import Validator_Tester
from agents.helpers.Evaluator import All_Evaluator
import wandb
import logging

class Sleep_Agent_Train():

    def __init__(self, config):

        self.config = config

        print_cuda_statistics()

        deterministic(self.config.training_params.seed)

        dataloader = globals()[self.config.dataset.dataloader_class]
        self.data_loader = dataloader(config=config)

        self.initialize_losses()
        self.initialize_logs()

        #Initialize Helpers
        self.mem_loader = Loader(agent = self)
        self.monitor_n_saver = Monitor_n_Save(agent = self)
        self.trainer = Trainer(agent = self)
        self.validator_tester = Validator_Tester(agent = self)
        self.evaluators = All_Evaluator(self.config, dataloaders=self.data_loader)

        self.mem_loader.load_models_n_optimizer()
        self.mem_loader.get_scheduler()

        wandb.watch(self.model, log_freq=100)
        logging.info("Available cuda devices: {}, current device:{}".format(torch. cuda. device_count(),torch.cuda.current_device()))


    def initialize_losses(self):
        self.device = "cuda:{}".format(self.config.training_params.gpu_device[0])

        self.loss = nn.CrossEntropyLoss()
        self.alignment_loss = nn.CrossEntropyLoss()
        self.alignment_target = torch.eye(n=500).unsqueeze(dim=0).repeat(500, 1, 1)[
                                :self.config.training_params.batch_size,
                                :self.config.dataset.outer_seq_length, :self.config.dataset.outer_seq_length
                                ].argmax(dim=-1).cuda()

    def initialize_logs(self):

        self.steps_no_improve = 0
        if self.config.early_stopping.validate_every:
            max_steps = int(len(self.data_loader.train_loader) / self.config.early_stopping.validate_every) + 1

            logging.info("Total update steps per epoch {}".format(len(self.data_loader.train_loader)))
            logging.info("Validate every {} update steps".format(self.config.early_stopping.validate_every))
            logging.info("Validation measures per epoch {}".format(max_steps))

        self.logs = {"current_epoch":0,"current_step":0,"steps_no_improve":0, "saved_step": 0, "train_logs":{},"val_logs":{},"test_logs":{},"best_logs":{"loss":{"total":100}, "acc":{"combined":0}} , "seed":self.config.training_params.seed}
        if self.config.training_params.wandb_disable:
            self.wandb_run = wandb.init(reinit=True, project="sleep_transformers", config=self.config, mode = "disabled", dir="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2021_data/wandb")
        else:
            self.wandb_run = wandb.init(reinit=True, project="sleep_transformers", config=self.config, dir="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2021_data/wandb" )

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.config.model.load_ongoing:
                self.mem_loader.sleep_load(self.config.model.save_dir)

            self.trainer.sleep_train()
            self.wandb_run.finish()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.validator_tester.validate()
        test_metrics = self.evaluators.val_evaluator.evaluate()
        self.monitor_n_saver.print_valid_results(test_metrics, current_step=self.logs["current_step"])
        self.validator_tester.validate(best_model = True, test_set = True)
        test_metrics = self.evaluators.test_evaluator.evaluate()
        self.monitor_n_saver.print_valid_results(test_metrics, current_step=self.logs["current_step"])
        self.monitor_n_saver.sleep_save(post_test_results = test_metrics)
        self.wandb_run.finish()



