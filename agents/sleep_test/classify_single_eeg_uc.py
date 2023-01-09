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
from graphs.models.custom_unet import EEG_CNN, EEG_CNN_2, EEG_CNN_3, EEG_CNN_4, EEG_CNN_5, EEG_CNN_6, EEG_UCnet_1, EEG_UCnet_2
from utils.misc import print_cuda_statistics
import copy
import time
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from datasets.sleepset import SleepDataLoader
# from pynvml import *

# from utils.lr_finders.lr_finder_eeg import LRFinder

cudnn.benchmark = True

class Sleep_Agent_Init_Train_EEG_UC(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        torch.backends.cudnn.enabled = False

        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)  # if you are using multi-GPU.
        np.random.seed(self.config.seed)  # Numpy module.
        random.seed(self.config.seed)  # Python random module.
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.data_loader = SleepDataLoader(config=config)

        # self.model = EEG_Unet_1(1,1)
        self.model = EEG_UCnet_2()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2), eps = 1e-08,
                                         weight_decay=self.config.weight_decay)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.max_epoch)
        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.config.learning_rate, max_lr=self.config.max_lr, cycle_momentum=False)

        # self.weights = torch.from_numpy(self.data_loader.weights).float()

        # initialize counter
        self.weights = torch.from_numpy(self.data_loader.weights).float()
        print("Available cuda devices: {}, current device:{}".format(torch. cuda. device_count(),torch.cuda.current_device()))
        # nvmlInit()
        # for i in self.config.gpu_device:
        #     h = nvmlDeviceGetHandleByIndex(i)
        #     info = nvmlDeviceGetMemoryInfo(h)
        #     print(f'total    : {info.total/(1024*1024)}')
        #     print(f'free     : {info.free/(1024*1024)}')
        #     print(f'used     : {info.used/(1024*1024)}')
        torch.cuda.manual_seed(self.config.seed)
        # self.device = "cuda:{}".format(self.config.gpu_device[0])
        self.device = torch.device(self.config.gpu_device[0])
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=[torch.device(i) for i in self.config.gpu_device])
        self.best_model = copy.deepcopy(self.model)

        self.loss = nn.CrossEntropyLoss(self.weights.to(self.device))
        self.loss2 = nn.MSELoss()

        self.current_epoch  = 0
        self.train_logs = torch.empty((self.config.max_epoch,12)).to(self.device)
        self.best_logs = torch.zeros(5).to(self.device)
        self.test_logs = torch.empty((self.config.max_epoch,7)).to(self.device)

        """
        Code to run the learning rate finder, be careful it might need some changes on dataloaders at the source code.
            
            lr_finder = LRFinder(self.model, self.optimizer, self.loss, device=self.device)
            lr_finder.range_test(self.data_loader.train_loader, end_lr=100, num_iter=100)
            _, lr = lr_finder.plot()    # to inspect the loss-learning rate graph
            lr_finder.reset()           # to reset the model and optimizer to their initial state
        """
        self.logger.info("Program will run on *****GPU-CUDA***** ")
        print_cuda_statistics()

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
            print("Models has saved successfully")
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
            print("Models has saved successfully")
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
                print("The best in epoch:", self.best_logs[4].item(), "so far acc:",self.best_logs[1].item()," and f1:",self.best_logs[2].item())

            test_loss, test_acc, test_f1, test_k, test_auc, test_conf, test_loss1, test_loss2 = self.test()
            print("Test accuracy: {0:.2f}% and f1: {1:.4f}".format(test_acc * 100, test_f1))
            print("Test loss1: {0:5f}% and loss2: {1:.5f}".format(test_loss1, test_loss2))
            print("Test kappa: {0:.4f}% and auc: {1:.4f}".format(test_k, test_auc))
            print("Test confusion matrix:")
            print(test_conf)

            self.train()

            # self.save_encoder()
            self.model = self.best_model

            self.save_encoder(self.config.save_dir_encoder)

            val_loss, val_acc, val_f1, val_k,_,_  = self.validate()
            print("Validation accuracy: {0:.2f}% and f1: {1:.4f} and k: {2:.4f}".format(val_acc*100, val_f1, val_k))
            test_loss, test_acc, test_f1, test_k, test_auc, test_conf, test_loss1, test_loss2 = self.test()
            print("Test accuracy: {0:.2f}% and f1: {1:.4f}".format(test_acc*100,test_f1))
            print("Test kappa: {0:.4f}% and auc: {1:.4f}".format(test_k,test_auc))
            print("Test confusion matrix:")
            print(test_conf)
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
            train_loss, train_acc, train_f1, train_k, train_loss1, train_loss2 = self.train_one_epoch()
            val_loss, val_acc, val_f1, val_k, val_loss1,val_loss2 = self.validate()
            self.train_logs[self.current_epoch] = torch.tensor([val_loss,train_loss,train_acc,val_acc,train_f1,val_f1,train_k,val_k,train_loss1, val_loss1, train_loss2, val_loss2],device=self.device)
            if self.config.validation:
                not_saved = True
                print("Epoch {0:d} Validation loss: {1:.6f}, accuracy: {2:.2f}% f1 :{3:.4f}, k :{4:.4f}  Training loss: {5:.6f}, accuracy: {6:.2f}% f1 :{7:.4f}, k :{8:.4f},".format(self.current_epoch, val_loss, val_acc*100, val_f1, val_k, train_loss, train_acc*100, train_f1, train_k))
                if (val_loss < self.best_logs[1].item()):
                    self.best_logs = torch.tensor([self.current_epoch, val_loss, val_acc, val_f1, val_k],device=self.device)
                    print("we have a new best at epoch {0:d} with validation accuracy: {1:.2f}%, f1: {2:.4f} and k: {3:.4f}".format(self.current_epoch, val_acc*100, val_f1, val_k))
                    self.best_model = copy.deepcopy(self.model)
                    test_loss, test_acc, test_f1, test_k, test_auc, test_conf, test_loss1, test_loss2  = self.test()
                    print("Test loss: {0:.6}, accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, auc :{4:.4f}".format(test_loss, test_acc*100, test_f1, test_k, test_auc))
                    self.test_logs[self.test_it] = torch.tensor([self.current_epoch,test_loss,test_acc,test_f1,test_k, test_loss1, test_loss2],device=self.device)
                    self.test_it+=1
                    self.save_checkpoint(self.config.save_dir)
                    not_saved = False
                    epochs_no_improve = 0
                else:
                    test_loss, test_acc, test_f1, test_k, test_auc, test_conf, test_loss1, test_loss2  = self.test()
                    # print("Test loss: {0:.6}, accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, auc :{4:.4f}".format(test_loss, test_acc*100, test_f1, test_k, test_auc))
                    self.test_logs[self.test_it] = torch.tensor([self.current_epoch,test_loss,test_acc,test_f1,test_k,test_loss1, test_loss2],device=self.device)
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
                print("Epoch {0:d} Validation loss: {1:.6f}, accuracy: {2:.2f}% f1 :{3:.4f}, k :{4:.4f}  Training loss: {5:.6f}, accuracy: {6:.2f}% f1 :{7:.4f}, k :{8:.4f},".format(self.current_epoch, val_loss, val_acc*100, val_f1, val_k, train_loss, train_acc*100, train_f1, train_k))
                self.best_model = copy.deepcopy(self.model)
                test_loss, test_acc, test_f1, test_k, test_auc, test_conf, test_loss1, test_loss2 = self.test()
                print("Test loss: {0:.6f}, accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, auc :{4:.4f}".format(test_loss, test_acc * 100, test_f1, test_k,test_auc))
                self.test_logs[self.test_it] = torch.tensor([self.current_epoch, test_loss, test_acc, test_f1, test_k, test_loss1, test_loss2], device=self.device)
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
        batch_loss, batch_loss1, batch_loss2 = 0,0,0
        tts, preds = [], []
        for batch_idx, (data, target, _) in tqdm(enumerate(self.data_loader.train_loader),"Training",leave=False, disable=self.config.tdqm): #enumerate(self.data_loader.train_loader):
            # self.plot_eeg(data[0].numpy())
            view_1, target = data[0].float().to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            pred, view = self.model(view_1)
            loss1 = self.loss(pred, target)
            loss2 = self.loss2(view, view_1)
            loss = loss1 + loss2
            loss.backward()
            batch_loss1 +=loss1.item()
            batch_loss2 +=loss2.item()
            batch_loss +=loss.item()
            tts.append(target)
            preds.append(pred)
            self.optimizer.step()
            self.scheduler.step()
        tts = torch.cat(tts).cpu().numpy()
        preds = torch.cat(preds).argmax(axis=1).cpu().numpy()
        return batch_loss/len(tts),  np.equal(tts,preds).sum()/len(tts), f1_score(preds,tts), cohen_kappa_score(preds,tts), batch_loss1/len(tts), batch_loss2/len(tts)

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        valid_loss,  batch_loss1, batch_loss2 = 0,0,0
        tts, preds = [], []
        with torch.no_grad():
            for batch_idx, (data, target, _) in tqdm(enumerate(self.data_loader.valid_loader),"Validation",leave=False, disable=self.config.tdqm): #enumerate(self.data_loader.valid_loader):
                view_1, target = data[0].float().to(self.device), target.to(self.device)
                pred, view = self.model(view_1)
                loss1 = self.loss(pred, target)
                loss2 = self.loss2(view, view_1)
                loss = loss1 + loss2
                batch_loss1 += loss1.item()
                batch_loss2 += loss2.item()
                valid_loss += loss.item()
                tts.append(target)
                preds.append(pred)
            tts = torch.cat(tts).cpu().numpy()
            preds = torch.cat(preds).cpu().numpy()
            for w_idx in range(0, len(preds), self.config.post_proc_step):
                for n_class in range(len(preds[0])):
                    preds[w_idx:w_idx + self.config.post_proc_step, n_class] = preds[ w_idx:w_idx + self.config.post_proc_step, n_class].sum() / self.config.post_proc_step
            preds = preds.argmax(axis=1)
        return valid_loss/len(tts), np.equal(tts, preds).sum()/len(tts), f1_score(preds, tts), cohen_kappa_score(tts, preds), batch_loss1/len(tts), batch_loss2/len(tts)

    def test(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        test_loss, batch_loss1, batch_loss2 = 0,0,0

        tts = []
        preds = []
        with torch.no_grad():
            for batch_idx, (data, target, _) in tqdm(enumerate(self.data_loader.test_loader),"Test",leave=False, disable=self.config.tdqm):
                view_1, target = data[0].float().to(self.device), target.to(self.device)
                pred, view = self.model(view_1)
                loss1 = self.loss(pred, target)
                loss2 = self.loss2(view, view_1)
                loss = loss1 + loss2
                batch_loss1 += loss1.item()
                batch_loss2 += loss2.item()
                test_loss += loss.item()
                tts.append(target)
                preds.append(pred)
            tts = torch.cat(tts).cpu().numpy()
            preds = torch.cat(preds).cpu().numpy()
            for w_idx in range(0, len(preds), self.config.post_proc_step):
                for n_class in range(len(preds[0])):
                    preds[w_idx:w_idx + self.config.post_proc_step, n_class] = preds[ w_idx:w_idx + self.config.post_proc_step, n_class].sum() / self.config.post_proc_step
            preds = preds.argmax(axis=1)
            test_acc = np.equal(tts, preds).sum() / len(tts)
            test_f1 = f1_score(preds, tts)
            test_k = cohen_kappa_score(tts, preds)
            test_auc = roc_auc_score(tts, preds)
            test_conf = confusion_matrix(tts, preds)
        return test_loss, test_acc, test_f1, test_k, test_auc, test_conf, batch_loss1/len(tts), batch_loss2/len(tts)

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        # self.save_checkpoint("./data/{}".format(self.config.checkpoint_file),0)
        print("We are in the final state.")
        self.plot_losses()
        self.plot_k()
        # self.save_checkpoint(self.config.save_dir)
        # print("test mse is {}".format(self.test()))

    def train_time_shift(self):
        """
        One epoch of training
        :return:
        """

        self.model.train()
        batch_loss = 0
        tts = []
        preds = []
        for batch_idx, (data, target, _) in enumerate(self.data_loader.train_loader): #tqdm(enumerate(self.data_loader.train_loader),"Training", leave=False):
            # self.plot_eeg(data[0].numpy())
            view_1 = data[0].unsqueeze(dim=-1).permute(0,3,1,2).float()
            view_1, target = view_1.to(self.device), target.to(self.device)

            if (batch_idx==0):
                past_view = view_1
            elif (batch_idx==1):
                self.optimizer.zero_grad()
                current_view = view_1
            elif (batch_idx==len(self.data_loader.train_loader)-1):
                break
            else:
                self.optimizer.zero_grad()
                future_view = view_1
                pred = self.model(past_view,current_view,future_view)
                loss = self.loss(pred, target)
                loss.backward()
                batch_loss +=loss
                tts.append(target.cpu().numpy())
                preds.append(pred.detach().cpu().numpy())
                self.optimizer.step()
                self.scheduler.step()
                past_view = current_view
                current_view = future_view
        tts = np.array([x for i in tts for x in i])
        preds = np.array([x for i in preds for x in i]).argmax(axis=1)
        acc = np.equal(tts,preds).sum()/len(tts)
        f1 = f1_score(preds,tts)
        return batch_loss, acc, f1

    def validate_time_shift(self):

        self.model.eval()
        valid_loss = 0
        tts = []
        preds = []
        with torch.no_grad():
            for batch_idx, (data, target, _) in tqdm(enumerate(self.data_loader.valid_loader),"Validation", leave=False):
                view_1 = data[0].unsqueeze(dim=-1).permute(0, 3, 1, 2).float()
                view_1, target = view_1.to(self.device), target.to(self.device)

                if (batch_idx == 0):
                    past_view = view_1
                elif (batch_idx == 1):
                    self.optimizer.zero_grad()
                    current_view = view_1
                elif (batch_idx == len(self.data_loader.valid_loader) - 1):
                    break
                else:
                    self.optimizer.zero_grad()
                    future_view = view_1
                    pred = self.model(past_view, current_view, future_view)
                    valid_loss += self.loss(pred, target).item()
                    tts.append(target.cpu().numpy())
                    preds.append(pred.cpu().numpy())
                    past_view = current_view
                    current_view = future_view
            tts = np.array([x for i in tts for x in i])
            preds = np.array([x for i in preds for x in i]).argmax(axis=1)
            acc = np.equal(tts, preds).sum() / len(tts)
            f1 = f1_score(preds, tts)
        return valid_loss, acc, f1

    def plot_losses(self):
        plt.figure()
        plt.plot(range(self.current_epoch), self.train_logs[0:self.current_epoch, 1].cpu().numpy(),"b",label="Train")
        plt.plot(range(self.current_epoch), self.train_logs[0:self.current_epoch, 0].cpu().numpy(),"r",label="Valid")
        plt.plot(self.test_logs[0:self.test_it, 0].cpu().numpy(), self.test_logs[0:self.test_it, 1].cpu().numpy(),"y",label="Test")
        plt.axvline(x=self.best_logs[0].item(), linestyle="--")
        plt.xlabel('Epochs')
        plt.ylabel('Loss Values')
        plt.title("Loss")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(range(self.current_epoch), self.train_logs[0:self.current_epoch, 8].cpu().numpy(),"b", label="Train_1")
        plt.plot(range(self.current_epoch), self.train_logs[0:self.current_epoch, 10].cpu().numpy(),"b--", label="Train_2")
        plt.plot(range(self.current_epoch), self.train_logs[0:self.current_epoch, 9].cpu().numpy(),"r", label="Valid_1")
        plt.plot(range(self.current_epoch), self.train_logs[0:self.current_epoch, 11].cpu().numpy(),"r--", label="Valid_2")
        plt.plot(self.test_logs[0:self.test_it, 0].cpu().numpy(), self.test_logs[0:self.test_it, 5].cpu().numpy(),"y",label="Test_1")
        plt.plot(self.test_logs[0:self.test_it, 0].cpu().numpy(), self.test_logs[0:self.test_it, 6].cpu().numpy(),"y--",label="Test_2")
        plt.axvline(x=self.best_logs[0].item(), linestyle="--")
        plt.xlabel('Epochs')
        plt.ylabel('Loss Values')
        plt.title("Losses")
        plt.legend()
        plt.show()

    def plot_k(self):
        plt.figure()
        plt.plot(range(self.current_epoch), self.train_logs[0:self.current_epoch, 6].cpu().numpy(),"b", label="Train")
        plt.plot(range(self.current_epoch), self.train_logs[0:self.current_epoch, 7].cpu().numpy(),"r", label="Valid")
        plt.plot(self.test_logs[0:self.test_it, 0].cpu().numpy(), self.test_logs[0:self.test_it, 4].cpu().numpy(),"y",
                 label="Test")
        plt.axvline(x=self.best_logs[0].item())
        plt.xlabel('Epochs')
        plt.ylabel("Cohen's Kappa")
        plt.title("Kappa")
        plt.legend()
        plt.show()

    def plot_eeg(self,data):
        time = np.arange(0,30-1/1500,30/1500)
        plt.figure("EEG Window")
        data = data.squeeze()
        for i in range(9):
            plt.subplot(9,1,i+1)
            plt.plot(time,data[0][i])
        plt.show()