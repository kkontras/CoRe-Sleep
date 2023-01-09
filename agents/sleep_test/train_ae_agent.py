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

from agents.sleep_test.sleep_agent import Sleep_Agent
from graphs.models.custom_unet import *
from utils.misc import print_cuda_statistics
import copy
import time
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from datasets.sleepset import SleepDataLoader
from utils.deterministic_pytorch import deterministic
from utils.lr_finders.lr_finder_eeg import LRFinder

cudnn.benchmark = True

class Sleep_Agent_AE_Train(Sleep_Agent):

    def __init__(self, config):
        super().__init__(config)
        deterministic(config.seed)
        self.data_loader = SleepDataLoader(config=config)
        enc = self.sleep_load_encoder()
        model_class = globals()[self.config.model_class]
        self.model = model_class(enc)

        # self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.config.learning_rate, rho=0.9, eps=1e-06, weight_decay=self.config.weight_decay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2), eps = 1e-08,
                                         weight_decay=self.config.weight_decay)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.max_epoch)
        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.config.learning_rate, max_lr=self.config.max_lr, cycle_momentum=False)

        self.weights = torch.from_numpy(self.data_loader.weights).float()
        print("Available cuda devices: {}, current device:{}".format(torch. cuda. device_count(),torch.cuda.current_device()))

        self.device = "cuda:{}".format(self.config.gpu_device[0])
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
        # lr_finder = LRFinder(self.model, self.optimizer, self.loss, device=self.device)
        # lr_finder.range_test(self.data_loader.train_loader, end_lr=10, num_iter=100)
        # _, lr = lr_finder.plot()  # to inspect the loss-learning rate graph
        # lr_finder.reset()

        # self.logger.info("Program will run on *****GPU-CUDA***** ")
        print_cuda_statistics()

    def initialize_logs(self):
        self.current_epoch = 0
        self.train_logs = torch.zeros((self.config.max_epoch, 8)).to(self.device)
        self.best_logs = torch.zeros(5).to(self.device)
        self.best_logs[1] = 5
        self.test_logs = torch.zeros((self.config.max_epoch, 5)).to(self.device)

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.config.load_ongoing:
                self.sleep_load(self.config.save_dir)
                print("The best in epoch:", self.best_logs[0].item(), "so far acc:",self.best_logs[2].item()," and f1:",self.best_logs[3].item(), " and kappa:",self.best_logs[4].item())
                a = self.test_logs[:, 4].argmax()
                print("Best test in epoch {0:.2f} with loss {1:.5f} accuracy: {2:.2f}% f1: {3:.4f} kappa: {4:.4f}".format( self.test_logs[a, 0], self.test_logs[a, 1], self.test_logs[a, 2], self.test_logs[a, 3], self.test_logs[a, 4]))

            self.train()
            if self.config.rec_test:
                a = self.test_logs[:,4].argmax()
                print("Best test in epoch {0:.2f} with loss {1:.5f} accuracy: {2:.2f}% f1: {3:.4f} kappa: {4:.4f}".format(self.test_logs[a,0],self.test_logs[a,1],self.test_logs[a,2],self.test_logs[a,3],self.test_logs[a,4]))
            print("Best valid in epoch {0:.2f} with loss {1:.5f} accuracy: {2:.2f}% f1: {3:.4f} kappa: {4:.4f}".format(self.best_logs[0],self.best_logs[1],self.best_logs[2],self.best_logs[3],self.best_logs[4]))
            self.model = self.best_model
            val_loss, val_acc, val_f1, val_k  = self.sleep_validate()
            print("Validation accuracy: {0:.2f}% and f1: {1:.4f} and k: {2:.4f}".format(val_acc*100, val_f1, val_k))
            test_loss, test_acc, test_f1, test_k, test_auc, test_conf = self.sleep_test()
            print("Test accuracy: {0:.2f}% and f1: {1:.4f}".format(test_acc*100,test_f1))
            print("Test kappa: {0:.4f}% and auc: {1:.4f}".format(test_k,test_auc))
            print("Test confusion matrix:")
            print(test_conf)
        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        print('we are training model normally')
        early_stop = False
        if not self.config.load_ongoing:
            self.best_model = copy.deepcopy(self.model)
        epochs_no_improve, early_stop = 0, False
        # self.best_logs[1] = 0
        for self.current_epoch in range(self.current_epoch, self.config.max_epoch):
            for param_group in self.optimizer.param_groups:
                lr =  param_group['lr']
            if self.config.verbose:
                print("We have learning rate: {0:.5f}".format(lr))
            start = time.time()
            train_loss, train_acc, train_f1, train_k = self.sleep_train_one_epoch()
            val_loss, val_acc, val_f1, val_k = self.sleep_validate()
            self.train_logs[self.current_epoch] = torch.tensor([val_loss,train_loss,train_acc,val_acc,train_f1,val_f1,train_k,val_k],device=self.device)
            if self.config.validation:
                not_saved = True
                if self.config.verbose:
                    print("Epoch {0:d} Validation loss: {1:.6f}, accuracy: {2:.2f}% f1 :{3:.4f}, k :{4:.4f}  Training loss: {5:.6f}, accuracy: {6:.2f}% f1 :{7:.4f}, k :{8:.4f},".format(self.current_epoch, val_loss, val_acc*100, val_f1, val_k, train_loss, train_acc*100, train_f1, train_k))
                if (val_loss < self.best_logs[1].item()):
                    self.best_logs = torch.tensor([self.current_epoch, val_loss, val_acc, val_f1, val_k],device=self.device)
                    print("we have a new best at epoch {0:d} with validation accuracy: {1:.2f}%, f1: {2:.4f} and k: {3:.4f}".format(self.current_epoch, val_acc*100, val_f1, val_k))
                    self.best_model = copy.deepcopy(self.model)
                    if self.config.rec_test:
                        test_loss, test_acc, test_f1, test_k, test_auc, test_conf = self.sleep_test()
                        print("Test loss: {0:.6}, accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, auc :{4:.4f}".format(test_loss, test_acc*100, test_f1, test_k, test_auc))
                        self.test_logs[ self.current_epoch ] = torch.tensor([self.current_epoch,test_loss,test_acc,test_f1,test_k],device=self.device)
                    self.sleep_save(self.config.save_dir)
                    not_saved = False
                    epochs_no_improve = 0
                else:
                    if self.config.rec_test:
                        test_loss, test_acc, test_f1, test_k, test_auc, test_conf = self.sleep_test()
                        print("Test loss: {0:.6}, accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, auc :{4:.4f}".format(test_loss, test_acc*100, test_f1, test_k, test_auc))
                        self.test_logs[ self.current_epoch ] = torch.tensor([self.current_epoch,test_loss,test_acc,test_f1,test_k],device=self.device)
                    epochs_no_improve += 1
                if (self.current_epoch % self.config.save_every == 0 and not_saved):
                    self.sleep_save(self.config.save_dir)
                if self.config.verbose:
                    print("This epoch took {} seconds".format(time.time() - start))
                if self.current_epoch > 5 and epochs_no_improve == self.config.n_epochs_stop:
                    print('Early stopping!')
                    early_stop = True
                    break
            else:
                print("Epoch {0:d} Validation loss: {1:.6f}, accuracy: {2:.2f}% f1 :{3:.4f}, k :{4:.4f}  Training loss: {5:.6f}, accuracy: {6:.2f}% f1 :{7:.4f}, k :{8:.4f},".format(self.current_epoch, val_loss, val_acc*100, val_f1, val_k, train_loss, train_acc*100, train_f1, train_k))
                self.best_model = copy.deepcopy(self.model)
                test_loss, test_acc, test_f1, test_k, test_auc, test_conf = self.sleep_test()
                print("Test loss: {0:.6f}, accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, auc :{4:.4f}".format(test_loss, test_acc * 100, test_f1, test_k,test_auc))
                self.test_logs[ self.current_epoch ] = torch.tensor([self.current_epoch, test_loss, test_acc, test_f1, test_k], device=self.device)
                self.sleep_save(self.config.save_dir)

        if early_stop:
            print("Early Stopping Occurred")

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        # self.save_checkpoint("./data/{}".format(self.config.checkpoint_file),0)
        print("We are in the final state.")
        self.sleep_plot_losses()
        self.sleep_plot_k()
        return self.best_logs[4]
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
            elif batch_idx==len(self.data_loader.train_loader)-1:
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