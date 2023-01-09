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
from graphs.models.custom_layers.eeg_encoders import *
from graphs.models.attention_models.windowFeature_base import *
from utils.misc import print_cuda_statistics
import copy
import time
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from datasets.sleepset import *
from utils.deterministic_pytorch import deterministic
from utils.lr_finders.lr_finder_eeg import LRFinder

cudnn.benchmark = True

class Sleep_Agent_Init_Train(Sleep_Agent):

    def __init__(self, config):
        super().__init__(config)

        # deterministic(config.seed)
        self.device = "cuda:{}".format(self.config.gpu_device[0])

        dataloader = globals()[self.config.dataloader_class]
        self.data_loader = dataloader(config=config)

        self.plot_stft()

        self.weights = torch.from_numpy(self.data_loader.weights).float()
        self.loss = nn.CrossEntropyLoss(self.weights.to(self.device))
        # self.loss = nn.CrossEntropyLoss()

        enc = self.sleep_load_encoder()
        model_class = globals()[self.config.model_class]
        self.model = model_class(enc, channel = self.config.channel)

        print(self.model)
        model_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total number of trainable parameters are: {}".format(model_total_params))
        # self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.config.learning_rate, rho=0.9, eps=1e-06, weight_decay=self.config.weight_decay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2), eps = 1e-07,
                                         weight_decay=self.config.weight_decay)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9)

        if self.config.lr_finder:

            lr_finder = LRFinder(self.model, self.optimizer, self.loss, device=self.device)
            lr_finder.range_test(self.data_loader.train_loader, end_lr=1, num_iter=100)
            _, lr = lr_finder.plot()  # to inspect the loss-learning rate graph
            print("Suggested learning rate is {}".format(lr))
            lr_finder.reset()

        if config.scheduler == "cyclic":
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.config.learning_rate, max_lr=self.config.max_lr, cycle_momentum=False)

        elif config.scheduler == "cosanneal":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.max_epoch)




        print("Available cuda devices: {}, current device:{}".format(torch. cuda. device_count(),torch.cuda.current_device()))


        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=[torch.device(i) for i in self.config.gpu_device])
        enc = self.sleep_load_encoder()
        model_class = globals()[self.config.model_class]
        self.best_model = model_class(enc, channel = self.config.channel)
        self.best_model = nn.DataParallel(model_class(enc), device_ids=[torch.device(i) for i in self.config.gpu_device])

        self.initialize_logs()


        """
        Code to run the learning rate finder, be careful it might need some changes on dataloaders at the source code.
            
            lr_finder = LRFinder(self.model, self.optimizer, self.loss, device=self.device)
            lr_finder.range_test(self.data_loader.train_loader, end_lr=100, num_iter=100)
            _, lr = lr_finder.plot()    # to inspect the loss-learning rate graph
            lr_finder.reset()           # to reset the model and optimizer to their initial state
        """


        # self.logger.info("Program will run on *****GPU-CUDA***** ")
        print_cuda_statistics()

    def initialize_logs(self):
        if self.config.validate_every:
            max_steps = int(len(self.data_loader.train_loader) / self.config.validate_every) + 1
            print(len(self.data_loader.train_loader))
            print(self.config.validate_every)
            print(max_steps)
        else:
            max_steps = 1
        self.logs = {"current_epoch":0,"current_step":0,"steps_no_improve":0,"train_logs":{},"val_logs":{},"test_logs":{},"best_logs":{"val_loss":100} , "seed":self.config.seed, }

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.config.load_ongoing:
                self.sleep_load(self.config.save_dir)
                self.steps_no_improve = 0
                print("The best in epoch:", self.logs["best_logs"]["epoch"], "so far acc:",self.logs["best_logs"]["accuracy"]," and f1:",self.logs["best_logs"]["f1"], " and kappa:",self.logs["best_logs"]["kappa"])

                a = self.test_logs[:, 4].argmax()
                print("Best test in epoch {0:.2f} with loss {1:.5f} accuracy: {2:.2f}% f1: {3:.4f} kappa: {4:.4f}".format( self.test_logs[a, 0], self.test_logs[a, 1], self.test_logs[a, 2], self.test_logs[a, 3], self.test_logs[a, 4]))

            # self.calculate_batch_balance()
            # self.train()
            self.sleep_train_step()
            if self.config.rec_test:
                a = self.test_logs[:,4].argmax()
                print("Best test in epoch {0:.2f} with loss {1:.5f} accuracy: {2:.2f}% f1: {3:.4f} kappa: {4:.4f}".format(self.test_logs[a,0],self.test_logs[a,1],self.test_logs[a,2],self.test_logs[a,3],self.test_logs[a,4]))
            print("Best valid in epoch {0:.2f} with loss {1:.5f} accuracy: {2:.2f}% f1: {3:.4f} kappa: {4:.4f}".format(self.best_logs[0],self.best_logs[1],self.best_logs[2],self.best_logs[3],self.best_logs[4]))
            self.model.load_state_dict(self.best_model.state_dict())
            val_loss, val_acc, val_f1, val_k, val_perclassf1  = self.sleep_validate()
            print("Validation accuracy: {0:.2f}% and f1: {1:.4f} and k: {2:.4f}".format(val_acc*100, val_f1, val_k))
            if self.config.use_test_set:
                test_loss, test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens = self.sleep_test()
                print("Test accuracy: {0:.2f}% and f1: {1:.4f}".format(test_acc*100,test_f1))
                print("Test kappa: {0:.4f}% and auc: {1:.4f}".format(test_k,test_auc))
                print("Test spec: {0:.4f}% and sens: {1:.4f}".format(test_spec, test_sens))
                print("Test confusion matrix:")
                print(test_conf)
                return test_acc, test_f1, test_k, list(test_perclass_f1)
            return val_acc, val_f1, val_k, list(val_perclassf1)

            print("{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f} {6:.4f}".format(test_loss,val_k,test_k,test_f1,test_auc, test_acc, self.test_logs[a,4]))
        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")
            return  0,0,0,[0,0,0,0,0]

    def save_encoder(self):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        for i, file_name in enumerate(self.config.save_dir_encoder):
            if self.config.savetrainedEncoder[i]:
                save_dict = {}
                savior = {}
                if hasattr(self.best_model.module, "encoder"):
                    savior["encoder_state_dict"] = self.best_model.module.encoder.state_dict()
                elif hasattr(self.best_model.module, "enc_{}".format(i)):
                    enc = getattr(self.best_model.module, "enc_{}".format(i))
                    savior["encoder_state_dict"] = enc.state_dict()
                save_dict.update(savior)
                try:
                    torch.save(save_dict, file_name)
                    print("Encoder saved successfully")
                except:
                    raise Exception("Problem in model saving")

    def train(self):
        """
        Main training loop
        :return:
        """
        print('we are training model normally')
        print(self.config.save_dir)

        if not self.config.load_ongoing:
            self.best_model.load_state_dict(self.model.state_dict())
        epochs_no_improve, early_stop = 0, False

        for self.logs["current_epoch"] in range(self.logs["current_epoch"], self.config.max_epoch):
            for param_group in self.optimizer.param_groups:
                lr =  param_group['lr']
            if self.config.verbose:
                print("We have learning rate: {0:.7f}".format(lr))
            self.start = time.time()
            train_loss, train_acc, train_f1, train_k = self.sleep_train_one_epoch()
            val_loss, val_acc, val_f1, val_k = self.sleep_validate()
            self.train_logs[self.current_epoch] = torch.tensor([val_loss,train_loss,train_acc,val_acc,train_f1,val_f1,train_k,val_k],device=self.device)
            early_stop = self.monitoring([train_loss, train_acc, train_f1, train_k], [val_loss, val_acc, val_f1, val_k])
            if early_stop: break
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

    # def train_time_shift(self):
    #     """
    #     One epoch of training
    #     :return:
    #     """
    #
    #     self.model.train()
    #     batch_loss = 0
    #     tts = []
    #     preds = []
    #     for batch_idx, (data, target, _) in enumerate(self.data_loader.train_loader): #tqdm(enumerate(self.data_loader.train_loader),"Training", leave=False):
    #         # self.plot_eeg(data[0].numpy())
    #         view_1 = data[0].unsqueeze(dim=-1).permute(0,3,1,2).float()
    #         view_1, target = view_1.to(self.device), target.to(self.device)
    #
    #         if (batch_idx==0):
    #             past_view = view_1
    #         elif (batch_idx==1):
    #             self.optimizer.zero_grad()
    #             current_view = view_1
    #         elif batch_idx==len(self.data_loader.train_loader)-1:
    #             break
    #         else:
    #             self.optimizer.zero_grad()
    #             future_view = view_1
    #             pred = self.model(past_view,current_view,future_view)
    #             loss = self.loss(pred, target)
    #             loss.backward()
    #             batch_loss +=loss
    #             tts.append(target.cpu().numpy())
    #             preds.append(pred.detach().cpu().numpy())
    #             self.optimizer.step()
    #             self.scheduler.step()
    #             past_view = current_view
    #             current_view = future_view
    #     tts = np.array([x for i in tts for x in i])
    #     preds = np.array([x for i in preds for x in i]).argmax(axis=1)
    #     acc = np.equal(tts,preds).sum()/len(tts)
    #     f1 = f1_score(preds,tts)
    #     return batch_loss, acc, f1
    #
    # def validate_time_shift(self):
    #
    #     self.model.eval()
    #     valid_loss = 0
    #     tts = []
    #     preds = []
    #     with torch.no_grad():
    #         for batch_idx, (data, target, _) in tqdm(enumerate(self.data_loader.valid_loader),"Validation", leave=False):
    #             view_1 = data[0].unsqueeze(dim=-1).permute(0, 3, 1, 2).float()
    #             view_1, target = view_1.to(self.device), target.to(self.device)
    #
    #             if (batch_idx == 0):
    #                 past_view = view_1
    #             elif (batch_idx == 1):
    #                 self.optimizer.zero_grad()
    #                 current_view = view_1
    #             elif (batch_idx == len(self.data_loader.valid_loader) - 1):
    #                 break
    #             else:
    #                 self.optimizer.zero_grad()
    #                 future_view = view_1
    #                 pred = self.model(past_view, current_view, future_view)
    #                 valid_loss += self.loss(pred, target).item()
    #                 tts.append(target.cpu().numpy())
    #                 preds.append(pred.cpu().numpy())
    #                 past_view = current_view
    #                 current_view = future_view
    #         tts = np.array([x for i in tts for x in i])
    #         preds = np.array([x for i in preds for x in i]).argmax(axis=1)
    #         acc = np.equal(tts, preds).sum() / len(tts)
    #         f1 = f1_score(preds, tts)
    #     return valid_loss, acc, f1

    def calculate_batch_balance(self):
        calc = []
        pbar = tqdm(enumerate(self.data_loader.train_loader), desc="Court labels", leave=False, disable=self.config.tdqm_disable)
        for batch_idx, (data, target, inits, idxs) in pbar:
            numpy_targets = target.cpu().numpy()
            uni_targets = np.unique(numpy_targets)
            calc.append(len(uni_targets))
            pbar.set_description("Count batch {0:d}/{1:d} ".format(batch_idx, len(
                self.data_loader.train_loader)))
            pbar.refresh()
        calc = np.array(calc)
        _, out = np.unique(calc, return_counts=True)
        print(out)

    def plot_stft(self):
        import matplotlib.pyplot as plt
        # data, target, init, id = point
        # point = ne,xt(iter(self.data_loader.train_loader))
        num_rows = self.config.plot_stft_num
        if num_rows ==0:
            return
        count_labels = {}
        for i in range(self.config.num_classes):
            count_labels[i]= num_rows
        labels = {0:"Wake",1:"N1",2:"N2",3:"N3",4:"REM"}
        num_cols = 5
        import matplotlib.gridspec as gridspec
        fig = plt.figure()
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(wspace=0.05, hspace=0.025)
        ax = [plt.subplot(gs[i]) for i in range(num_rows * num_cols)]

        for data, target, init, id in self.data_loader.train_loader:
            for i in range(len(data[0])):
                for j in range(len(data[0][0])):
                    Zxx = data[0][i,j]
                    label = target[i,j].numpy()
                    if count_labels[int(label)]>0:
                        #Version_1
                        # Zxx = Zxx[0,:,0,:].numpy().transpose(1,0)
                        #Version_3
                        Zxx = Zxx[0,0,:,:].numpy()

                        f,t= Zxx.shape
                        t = np.arange(0,t)
                        f = np.arange(0,f)

                        PCM = ax[5*(count_labels[int(label)]-1)+int(label)].pcolormesh(t, f, Zxx, vmin=Zxx.min(), vmax=Zxx.max(), shading='gouraud')
                        if(count_labels[int(label)]==1):
                            ax[5*(count_labels[int(label)]-1)+int(label)].title.set_text('{}'.format(labels[int(label)]))
                        ax[5*(count_labels[int(label)]-1)+int(label)].axis("off")
                        count_labels[int(label)] -= 1
                        print(5*(count_labels[int(label)]-1)+int(label))
            c = 0
            for i in range(5):
                c += count_labels[int(i)]
            if c==0:
                break
        fig.text(0.5, 0.04, 'Time [sec]', ha='center')
        fig.text(0.04, 0.5, 'Frequency [Hz]', va='center', rotation='vertical')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # fig.colorbar(PCM, cax = ax)
        plt.show()


