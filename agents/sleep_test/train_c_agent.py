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
from graphs.models.u2net import U2NET
from utils.misc import print_cuda_statistics
import copy
import time
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from datasets.sleepset import *
from utils.deterministic_pytorch import deterministic
from utils.lr_finders.lr_finder_eeg import LRFinder
from utils.optimizers.lamp_optimizer import Lamb
from utils.schedulers.no_scheduler import No_Scheduler
from utils.loss.KD_Loss import KD_Loss
from graphs.losses.SCELoss import SCELoss
from agents.sleep_test.helpers.Loader import Loader
from agents.sleep_test.helpers.Monitor_n_Save import Monitor_n_Save
from agents.sleep_test.helpers.Trainer import Trainer
from agents.sleep_test.helpers.Validator_Tester import Validator_Tester
import wandb
from utils.loss.VAE_Loss import VAE_Loss

os.environ["WANDB_SILENT"] = "true"

cudnn.benchmark = True

class Sleep_Agent_Init_Train():

    def __init__(self, config, trial=None):
        # super().__init__(config)
        self.config = config

        deterministic(self.config.training_params.seed)

        self.device = "cuda:{}".format(self.config.training_params.gpu_device[0])
        # print_cuda_statistics()

        dataloader = globals()[self.config.dataset.dataloader_class]
        self.data_loader = dataloader(config=config)
        self.weights = torch.from_numpy(self.data_loader.weights).float()

        train_bad_perf_eeg_patients = [int(i) for i in['0008', '0035', '0065', '0095', '0109', '0111', '0128', '0152', '0160', '0186', '0200', '0240', '0264', '0270',
         '0271', '0275', '0409', '0461', '0468', '0496', '0527', '0584', '0658', '0746', '0995', '0996', '0997', '1048',
         '1133', '1255', '1325', '1345', '1351', '1352', '1357', '1394', '1428', '1432', '1449', '1450', '1454', '1474',
         '1500', '1501', '1548', '1566', '1587', '1597', '1638', '1682', '1696', '1706', '1709', '1732', '1822', '1874',
         '1950', '2040', '2041', '2060', '2092', '2132', '2155', '2176', '2223', '2232', '2254', '2277', '2278', '2302',
         '2332', '2372', '2447', '2470', '2504', '2543', '2601', '2607', '2610', '2619', '2684', '2715', '2769', '2851',
         '2879', '2938', '2944', '2946', '3044', '3090', '3099', '3103', '3107', '3114', '3145', '3163', '3173', '3191',
         '3200', '3215', '3228', '3256', '3366', '3409', '3631', '3633', '3735', '3745', '3766', '3815', '3886', '3951',
         '4071', '4174', '4214', '4224', '4254', '4398', '4405', '4448', '4482', '4560', '4699', '4815', '4820', '4898',
         '4937', '4970', '4999', '5068', '5098', '5134', '5154', '5226', '5253', '5316', '5361', '5371', '5375', '5508',
         '5534', '5544', '5697', '5729', '5762']]

        # self.data_loader.train_loader.dataset.choose_specific_patient(train_bad_perf_eeg_patients, include_chosen=False)

        # self.plot_stft()
        # self.plot_signals()

        self.initialize_logs()

        if "kd_label" in self.config.dataset and self.config.dataset.kd_label:
            self.loss = KD_Loss(alpha=0.3, temp=1)
        else:
            if "softlabels" in self.config.dataset and self.config.dataset.softlabels:
                self.loss = nn.BCEWithLogitsLoss(weight=self.weights.to(self.device))
            elif "sceloss" in self.config and self.config.sceloss.use:
                self.loss = SCELoss(alpha=self.config.sceloss.alpha, beta=self.config.sceloss.beta)
            else:
                self.loss = nn.CrossEntropyLoss(self.weights.to(self.device))

        print(self.loss.weight)

        if "training_type" in self.config.model.args and (self.config.model.args.training_type == "alignment" or self.config.model.args.training_type == "alignment_order" or self.config.model.args.training_type == "alignment_order_multisupervised"):

            if "blip_loss" in self.config.model.args.multi_loss:
                self.blip_loss = nn.BCEWithLogitsLoss(weight=self.weights.to(self.device))
                self.blip_target = 0
                self.order_loss = nn.CrossEntropyLoss()
                self.alignment_loss = nn.BCEWithLogitsLoss(weight=self.weights.to(self.device))
                self.alignment_target = torch.eye(n=500).unsqueeze(dim=0).repeat(500, 1, 1)[:self.config.training_params.batch_size,
                                   :self.config.dataset.seq_length[0], :self.config.dataset.seq_length[0]].cuda()
            else:
                self.alignment_loss = nn.CrossEntropyLoss()
                self.order_loss = nn.CrossEntropyLoss()
                # self.consistency_loss = nn.KLDivLoss(reduction="batchmean")
                self.consistency_loss = nn.MSELoss()
                self.reconstruction_loss = VAE_Loss()
                if self.config.model.args.training_type == "alignment_order": self.order_loss = nn.CrossEntropyLoss()
                self.alignment_target = torch.eye(n=500).unsqueeze(dim=0).repeat(500, 1, 1)[:self.config.training_params.batch_size,
                                   :self.config.dataset.seq_length[0], :self.config.dataset.seq_length[0]].argmax(dim=-1).cuda()

        self.mem_loader = Loader(agent = self)
        self.monitor_n_saver = Monitor_n_Save(agent = self)
        self.trainer = Trainer(agent = self)
        self.validator_tester = Validator_Tester(agent = self)

        self.mem_loader.load_models_n_optimizer()
        self.mem_loader.get_scheduler()

        wandb.watch(self.model, log_freq=100)

        # print(self.model)


        if self.config.lr_finder:
            print("This needs to refined")

            lr_finder = LRFinder(self.model, self.optimizer, self.loss, device=self.device)
            lr_finder.range_test(self.data_loader.train_loader, val_loader=self.data_loader.valid_loader, end_lr=1, num_iter=self.config.lr_finder_steps)
            _, lr = lr_finder.plot()  # to inspect the loss-learning rate graph
            print("Suggested learning rate is {}".format(lr))
            lr_finder.reset()
            del self.optimizer
            self.optimizer = Lamb(self.model.parameters(), lr=lr, betas=(self.config.beta1, self.config.beta2), eps = 1e-07,
                                             weight_decay=self.config.weight_decay)

        print("Available cuda devices: {}, current device:{}".format(torch. cuda. device_count(),torch.cuda.current_device()))



        """
        Code to run the learning rate finder, be careful it might need some changes on dataloaders at the source code.
            
            lr_finder = LRFinder(self.model, self.optimizer, self.loss, device=self.device)
            lr_finder.range_test(self.data_loader.train_loader, end_lr=100, num_iter=100)
            _, lr = lr_finder.plot()    # to inspect the loss-learning rate graph
            lr_finder.reset()           # to reset the model and optimizer to their initial state
        """


        # self.logger.info("Program will run on *****GPU-CUDA***** ")

    def initialize_logs(self):

        self.steps_no_improve = 0
        if self.config.early_stopping.validate_every:
            max_steps = int(len(self.data_loader.train_loader) / self.config.early_stopping.validate_every) + 1
            print(len(self.data_loader.train_loader))
            print(self.config.early_stopping.validate_every)
            print(max_steps)
        else:
            max_steps = 1

        self.logs = {"current_epoch":0,"current_step":0,"steps_no_improve":0, "saved_step": 0, "train_logs":{},"val_logs":{},"test_logs":{},"best_logs":{"val_loss":{"total":100}} , "seed":self.config.training_params.seed, "weights": self.weights}
        if self.config.training_params.wandb_disable:
            self.wandb_run = wandb.init(reinit=True, project="sleep_transformers", config=self.config, mode = "disabled", dir="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2021_data/wandb")
        else:
            self.wandb_run = wandb.init(reinit=True, project="sleep_transformers", config=self.config, dir="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2021_data/wandb" )

    def run(self, trial=None):
        """
        The main operator
        :return:
        """
        try:
            if self.config.model.load_ongoing:
                self.mem_loader.sleep_load(self.config.model.save_dir)

            # self.calculate_batch_balance()
            # self.train()

            # self.get_teacher_estimations()

            self.trainer.sleep_train_step(trial)

            if self.config.training_params.rec_test:
                a = self.test_logs[:,4].argmax()
                print("Best test in epoch {0:.2f} with loss {1:.5f} accuracy: {2:.2f}% f1: {3:.4f} kappa: {4:.4f}".format(self.test_logs[a,0],self.test_logs[a,1],self.test_logs[a,2],self.test_logs[a,3],self.test_logs[a,4]))
            # print("Best valid in epoch {0:.2f} with loss {1:.5f} accuracy: {2:.2f}% f1: {3:.4f} kappa: {4:.4f}".format(self.best_logs[0],self.best_logs[1],self.best_logs[2],self.best_logs[3],self.best_logs[4]))
            self.model.load_state_dict(self.best_model.state_dict())
            val_metrics  = self.validator_tester.sleep_validate()
            print(val_metrics)
            self.wandb_run.finish()
            # return val_loss["total"]

            # print("{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f} {6:.4f}".format(test_loss,val_k,test_k,test_f1,test_auc, test_acc, self.test_logs[a,4]))
        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")
            return  0,0,0,[0,0,0,0,0]

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        # self.save_checkpoint("./data/{}".format(self.config.checkpoint_file),0)
        print("We are in the final state.")
        return self.logs["best_logs"]["val_loss"]["total"]
        # self.sleep_plot_losses()
        # self.sleep_plot_k()
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
        num_rows = self.config.plot_stft_num * len(self.config.data_view_dir)
        if num_rows ==0:
            return
        count_labels = {}
        for i in range(self.config.num_classes):
            count_labels[i]= self.config.plot_stft_num
        # labels = {0:"Wake",1:"N1",2:"N2",3:"N3",4:"REM"}
        labels = {0:"Non-Seizure",1:"Seizure",2:"N2",3:"N3",4:"REM"}
        num_cols = self.config.num_classes
        import matplotlib.gridspec as gridspec
        fig = plt.figure()
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(wspace=0.05, hspace=0.025)
        ax = [plt.subplot(gs[i]) for i in range(num_rows * num_cols)]
        for data, target, init, id in self.data_loader.train_loader:
            for i in range(len(data[0])):
                for j in range(len(data[0][0])):
                    eeg = data[0][i,j]
                    ecg = data[1][i,j]
                    label = target[i,j].numpy()
                    if count_labels[int(label)]>0:
                        lottery_ticket = np.random.randint(4)
                        if lottery_ticket > 0:
                            # Give me kind of random plots, not always the same
                            continue
                        #Version_1
                        # Zxx = Zxx[0,:,0,:].numpy().transpose(1,0)
                        #Version_3

                        eeg = eeg[0,:,:,:].numpy()
                        ecg = ecg[0,:,:,:].numpy()

                        t = np.arange(0,eeg.shape[-1])
                        f = np.arange(0,eeg.shape[-2])
                        num_of_plot = (self.config.num_classes + len(self.config.data_view_dir)) * (count_labels[int(label)] - 1) + int(label)

                        # for signal in eeg:
                        signal = eeg[0]
                        _ = ax[num_of_plot].pcolormesh(t, f, signal, vmin=signal.min(), vmax=signal.max(), shading='gouraud')
                        t = np.arange(0, ecg.shape[-1])
                        f = np.arange(0, ecg.shape[-2])
                        # for signal in ecg:
                        signal = ecg[0]
                        _ = ax[num_of_plot+2].pcolormesh(t, f, signal, vmin=signal.min(), vmax=signal.max(), shading='gouraud')

                        if(count_labels[int(label)]==1):
                            ax[self.config.num_classes*(count_labels[int(label)]-1)+int(label)].title.set_text('{}'.format(labels[int(label)]))


                        if label == 0:
                            ax[num_of_plot].set_yticks([0])
                            ax[num_of_plot].set_yticklabels(["EEG"])
                            ax[num_of_plot].set_xticks([])
                            ax[num_of_plot + 2].set_yticks([0])
                            ax[num_of_plot + 2].set_xticks([])
                            ax[num_of_plot + 2].set_yticklabels(["ECG"])
                            [s.set_visible(False) for s in ax[num_of_plot].spines.values()]
                            [s.set_visible(False) for s in ax[num_of_plot + 2].spines.values()]
                            # ax[num_of_plot].spines['top'].set_visible(False)
                            # ax[num_of_plot+2].spines['top'].set_visible(False)
                        else:
                            ax[num_of_plot].set_xticks([])
                            ax[num_of_plot].set_yticks([])
                            ax[num_of_plot + 2].set_xticks([])
                            ax[num_of_plot + 2].set_yticks([])
                            [s.set_visible(False) for s in ax[num_of_plot].spines.values()]
                            [s.set_visible(False) for s in ax[num_of_plot + 2].spines.values()]

                        if num_of_plot == 0 or num_of_plot == 1:
                            ax[num_of_plot + 2].spines['bottom'].set_visible(True)
                        elif num_of_plot == 5 or num_of_plot == 4:
                            ax[num_of_plot].spines['top'].set_visible(True)
                        count_labels[int(label)] -= 1

            c = 0
            for i in range(self.config.num_classes):
                c += count_labels[int(i)]
            if c==0:
                break
        fig.text(0.5, 0.04, 'Time [sec]', ha='center')
        fig.text(0.04, 0.5, 'Frequency [Hz]', va='center', rotation='vertical')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # fig.colorbar(PCM, cax = ax)
        plt.show()

    def plot_signals(self):
        import matplotlib.pyplot as plt
        # data, target, init, id = point
        # point = ne,xt(iter(self.data_loader.train_loader))
        num_rows = len(self.config.data_view_dir)*self.config.plot_signals_num
        if num_rows ==0:
            return
        count_labels = {}
        for i in range(self.config.num_classes):
            count_labels[i]= self.config.plot_stft_num
        # labels = {0:"Wake",1:"N1",2:"N2",3:"N3",4:"REM"}
        labels = {0:"Non-Epilepsy",1:"Epilepsy"}
        num_cols = self.config.num_classes
        import matplotlib.gridspec as gridspec
        fig = plt.figure()
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(wspace=0.05, hspace=0.025)
        ax = [plt.subplot(gs[i]) for i in range(num_rows * num_cols)]

        for data, target, init, id in self.data_loader.train_loader:
            for i in range(len(data[0])):
                # for j in range(len(data[0][0])):
                eeg = data[0][i,0]
                ecg = data[1][i,0]
                label = int(target[i].numpy())
                if count_labels[int(label)]>0:
                    lottery_ticket = np.random.randint(4)
                    if lottery_ticket >0 :
                        #Give me kind of random plots, not always the same
                        continue
                    #Version_1
                    # Zxx = Zxx[0,:,0,:].numpy().transpose(1,0)
                    #Version_3
                    eeg = eeg[0].numpy()
                    ecg = ecg[0].numpy()
                    print(eeg.shape)
                    print(ecg.shape)
                    t = np.arange(0,eeg.shape[1])
                    num_of_plot = (self.config.num_classes+len(self.config.data_view_dir))*(count_labels[int(label)]-1)+int(label)
                    for signal in eeg:
                        ax[num_of_plot].plot(t,signal)

                    print("EEG on {}".format(num_of_plot))
                    t = np.arange(0, ecg.shape[1])
                    for signal in ecg:
                        ax[num_of_plot+2].plot(t, signal)
                    print("ECG on {}".format(num_of_plot+2))
                    if(count_labels[label]==1):
                        ax[num_of_plot].title.set_text('{}'.format(labels[int(label)]))

                    if label == 0:
                        ax[num_of_plot].set_yticks([0])
                        ax[num_of_plot].set_yticklabels(["EEG"])
                        ax[num_of_plot].set_xticks([])
                        ax[num_of_plot+2].set_yticks([0])
                        ax[num_of_plot+2].set_xticks([])
                        ax[num_of_plot+2].set_yticklabels(["ECG"])
                        [s.set_visible(False) for s in ax[num_of_plot].spines.values()]
                        [s.set_visible(False) for s in ax[num_of_plot+2].spines.values()]
                        # ax[num_of_plot].spines['top'].set_visible(False)
                        # ax[num_of_plot+2].spines['top'].set_visible(False)
                    else:
                        ax[num_of_plot].set_xticks([])
                        ax[num_of_plot].set_yticks([])
                        ax[num_of_plot+2].set_xticks([])
                        ax[num_of_plot+2].set_yticks([])
                        [s.set_visible(False) for s in ax[num_of_plot].spines.values()]
                        [s.set_visible(False) for s in ax[num_of_plot+2].spines.values()]

                    if num_of_plot == 0 or num_of_plot == 1:
                        ax[num_of_plot + 2].spines['bottom'].set_visible(True)
                    elif num_of_plot == 5 or num_of_plot == 4:
                        ax[num_of_plot ].spines['top'].set_visible(True)
                    # ax[(self.config.num_classes+len(self.config.data_view_dir))*(count_labels[int(label)]-1)+int(label)].axis("off")
                    # ax[(self.config.num_classes+len(self.config.data_view_dir))*(count_labels[int(label)]-1)+int(label)+2].axis("off")
                    count_labels[int(label)] -= 1

            c = 0
            for i in range(self.config.num_classes):
                c += count_labels[int(i)]
            if c==0:
                break
        fig.text(0.5, 0.04, 'Time [sec]', ha='center')
        # fig.text(0.04, 0.5, 'Frequency [Hz]', va='center', rotation='vertical')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # fig.colorbar(PCM, cax = ax)
        plt.show()

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

        if self.config.training_params.rec_test:
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

        if self.config.training_params.rec_test:
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

        if self.config.training_params.rec_test:
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


