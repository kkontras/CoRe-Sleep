"""
Mnist Main agent, as mentioned in the tutorial
"""
import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
import torch.optim as optim

from agents.base import BaseAgent
from graphs.models.custom_unet import *
from utils.misc import print_cuda_statistics
import copy
import time
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from datasets.sleepset import SleepDataLoader
from utils.deterministic_pytorch import deterministic

# from utils.lr_finders.lr_finder_eeg import LRFinder

cudnn.benchmark = True

class Sleep_Agent_Init_Train_EEG_U(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        deterministic(config.seed)

        self.data_loader = SleepDataLoader(config=config)

        self.model = EEG_Unet(self.config.model_type)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2), eps = 1e-08,
                                         weight_decay=self.config.weight_decay)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.max_epoch)
        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.config.learning_rate, max_lr=self.config.max_lr, cycle_momentum=False)

        # self.weights = torch.from_numpy(self.data_loader.weights).float()

        # initialize counter
        self.weights = torch.from_numpy(self.data_loader.weights).float()
        print("Available cuda devices: {}, current device:{}".format(torch. cuda. device_count(),torch.cuda.current_device()))

        torch.cuda.manual_seed(self.config.seed)
        # self.device = "cuda:{}".format(self.config.gpu_device[0])
        self.device = torch.device(self.config.gpu_device[0])
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=[torch.device(i) for i in self.config.gpu_device])
        self.best_model = copy.deepcopy(self.model)

        self.loss = nn.L1Loss()
        self.initialize_logs()

        """
        Code to run the learning rate finder, be careful it might need some changes on dataloaders at the source code.
            
            lr_finder = LRFinder(self.model, self.optimizer, self.loss, device=self.device)
            lr_finder.range_test(self.data_loader.train_loader, end_lr=100, num_iter=100)
            _, lr = lr_finder.plot()    # to inspect the loss-learning rate graph
            lr_finder.reset()           # to reset the model and optimizer to their initial state
        """
        print_cuda_statistics()

    def initialize_logs(self):
        self.current_epoch  = 0
        self.train_logs = torch.empty((self.config.max_epoch,2)).to(self.device)
        self.best_logs = torch.zeros(2).to(self.device)
        self.test_logs = torch.empty((self.config.max_epoch,2)).to(self.device)

    def load_checkpoint(self, file_name):
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
        self.train_logs[0:checkpoint["train_logs"].shape[0],:] = checkpoint["train_logs"]
        self.test_logs[0:checkpoint["test_logs"].shape[0],:] = checkpoint["test_logs"]
        self.current_epoch = checkpoint["epoch"]
        self.best_logs = checkpoint["best_logs"]
        print("Model has loaded successfully")


    def save_checkpoint(self, file_name="checkpoint.pth.tar"):
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
        savior["train_logs"] = self.train_logs
        savior["test_logs"] = self.test_logs
        savior["epoch"] = self.current_epoch
        savior["best_logs"] = self.best_logs
        save_dict.update(savior)
        try:
            torch.save(save_dict, file_name)
            print("Models saved successfully")
        except:
            raise Exception("Problem in model saving")

    def save_encoder(self, file_name="checkpoint_encoder.pth.tar"):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        save_dict = {}
        savior = {}
        savior["encoder_state_dict"] = self.best_model.module.encoder.state_dict()
        save_dict.update(savior)
        try:
            torch.save(save_dict, file_name)
            print("Encoder saved successfully")
        except:
            raise Exception("Problem in model saving")


    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.config.load_ongoing:
                self.load_checkpoint(self.config.save_dir)
                print("The best in epoch:", self.best_logs[0].item(), "so far val loss:",self.best_logs[1].item())

            test_loss= self.test()
            print("Test loss: {0:.6f} ".format(test_loss))

            self.train()

            # self.save_encoder()
            self.model = self.best_model

            self.save_encoder(self.config.save_dir_encoder)

            val_loss = self.validate()
            print("Validation loss: {0:.6f}%".format(val_loss))
            test_loss = self.test()
            print("Test loss: {0:.6f} ".format(test_loss ))

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        print('we are training model normally')
        if not self.config.load_ongoing:
            self.best_model = copy.deepcopy(self.model)
        epochs_no_improve, early_stop = 0, False
        self.best_logs[1] = 1000
        self.test_it = self.current_epoch
        for self.current_epoch in range(self.current_epoch, self.config.max_epoch):
            for param_group in self.optimizer.param_groups:
                lr =  param_group['lr']
            print("We have learning rate: {0:.5f}".format(lr))
            start = time.time()
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            self.train_logs[self.current_epoch] = torch.tensor([val_loss,train_loss],device=self.device)
            if self.config.validation:
                not_saved = True
                print("Epoch {0:d} Validation loss: {1:.6f}  Training loss: {2:.6f}".format(self.current_epoch, val_loss, train_loss))
                if (val_loss < self.best_logs[1].item()):
                    self.best_logs = torch.tensor([self.current_epoch, val_loss],device=self.device)
                    print("we have a new best at epoch {0:d} with validation loss: {1:.6f}%".format(self.current_epoch, val_loss))
                    self.best_model = copy.deepcopy(self.model)
                    test_loss = self.test()
                    print("Test loss: {0:.6}".format(test_loss))
                    self.test_logs[self.test_it] = torch.tensor([self.current_epoch,test_loss],device=self.device)
                    self.test_it+=1
                    self.save_checkpoint(self.config.save_dir)
                    not_saved = False
                    epochs_no_improve = 0
                else:
                    test_loss = self.test()
                    # print("Test loss: {0:.6},:.6f}".format(test_loss))
                    self.test_logs[self.test_it] = torch.tensor([self.current_epoch,test_loss],device=self.device)
                    self.test_it+=1
                    epochs_no_improve += 1
                if (self.current_epoch % self.config.save_every == 0 and not_saved):
                    self.save_checkpoint(self.config.save_dir)
                print("This epoch took {} seconds".format(time.time() - start))
                if self.current_epoch > 5 and epochs_no_improve == self.config.n_epochs_stop:
                    print('Early stopping!')
                    early_stop = True
                    break
            else:
                print("Epoch {0:d} Validation loss: {1:.6f}  Training loss: {2:.6f}".format(self.current_epoch, val_loss, train_loss))
                self.best_model = copy.deepcopy(self.model)
                test_loss = self.test()
                print("Test loss: {0:.6},:.6f}".format(test_loss))
                self.test_logs[self.test_it] = torch.tensor([self.current_epoch, test_loss], device=self.device)
                self.test_it += 1
                self.save_checkpoint(self.config.save_dir)

        if early_stop:
            print("Early Stopping Occurred")

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """

        self.model.train()
        batch_loss, sum= 0,0
        for batch_idx, (data, target, _) in tqdm(enumerate(self.data_loader.train_loader),"Training",leave=False, disable=self.config.tdqm): #enumerate(self.data_loader.train_loader):
            # self.plot_eeg(data[0].numpy())
            self.optimizer.zero_grad()
            view_1, _ = data[0].float().to(self.device), target.to(self.device)
            pred = self.model(view_1)
            loss = self.loss(pred, view_1)
            loss.backward()
            batch_loss +=loss.item()
            sum += data[0].shape[0]
            self.optimizer.step()
            self.scheduler.step()
        return batch_loss/sum

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        valid_loss, sum= 0,0
        with torch.no_grad():
            for batch_idx, (data, target, _) in tqdm(enumerate(self.data_loader.valid_loader),"Validation",leave=False, disable=self.config.tdqm): #enumerate(self.data_loader.valid_loader):
                view_1, _ = data[0].float().to(self.device), target.to(self.device)
                pred = self.model(view_1)
                loss = self.loss(pred, view_1)
                valid_loss += loss.item()
                sum += data[0].shape[0]
        return valid_loss/sum

    def test(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        test_loss, sum= 0,0
        with torch.no_grad():
            for batch_idx, (data, target, _) in tqdm(enumerate(self.data_loader.test_loader),"Test",leave=False, disable=self.config.tdqm):
                view_1, _ = data[0].float().to(self.device), target.to(self.device)
                pred = self.model(view_1)
                loss = self.loss(pred, view_1)
                test_loss += loss.item()
                sum += data[0].shape[0]
        # self.plot_segment(view_1.cpu().numpy()[0],pred.cpu().numpy()[0])
        return test_loss/sum

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        # self.save_checkpoint("./data/{}".format(self.config.checkpoint_file),0)
        print("We are in the final state.")
        self.plot_loss()
        # self.save_checkpoint(self.config.save_dir)
        # print("test mse is {}".format(self.test()))

    def plot_loss(self):
        plt.figure()
        plt.plot(range(self.current_epoch), self.train_logs[0:self.current_epoch, 1].cpu().numpy(), label="Train")
        plt.plot(range(self.current_epoch), self.train_logs[0:self.current_epoch, 0].cpu().numpy(), label="Valid")
        plt.plot((self.best_logs[0].item(), self.best_logs[0].item()), (0, self.best_logs[1].item()), linestyle="--",
                 color="y", label="Chosen Point")
        plt.plot((0, self.best_logs[0].item()), (self.best_logs[1].item(), self.best_logs[1].item()), linestyle="--",
                 color="y")

        if self.config.rec_test:
            plt.plot(range(self.current_epoch), self.test_logs[0:self.current_epoch, 1].cpu().numpy(), label="Test")
            best_test = self.test_logs[:, 1].argmin()
            plt.plot((self.test_logs[best_test, 0].item(), self.test_logs[best_test, 0].item()),
                     (0, self.test_logs[best_test, 1].item()), linestyle="--", color="r", label="Actual Best Loss")
            plt.plot((0, self.test_logs[best_test, 0].item()),
                     (self.test_logs[best_test, 1].item(), self.test_logs[best_test, 1].item()), linestyle="--",
                     color="r")

        plt.xlabel('Epochs')
        plt.ylabel('Loss Values')
        plt.title("Loss")
        plt.legend()
        plt.show()

    def plot_segment(self, segment1, pred):
        time = np.arange(0, 30 - 1 / 900, 30 / 900)
        for i in range(3):
            plt.subplot(3,1,i+1)
            plt.plot(time,segment1[0][i])
            plt.plot(time,pred[0][i])
        plt.show()

    def plot_eeg(self,data):
        time = np.arange(0,30-1/1500,30/1500)
        plt.figure("EEG Window")
        data = data.squeeze()
        for i in range(9):
            plt.subplot(9,1,i+1)
            plt.plot(time,data[0][i])
        plt.show()