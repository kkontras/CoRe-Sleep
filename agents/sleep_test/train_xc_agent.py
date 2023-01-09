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
from datasets.sleepset import SleepDataLoader
from utils.deterministic_pytorch import deterministic
from utils.lr_finders.lr_finder_eeg import LRFinder

cudnn.benchmark = True

class Sleep_Agent_Init_Train_X(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        deterministic(config.seed)
        self.device = "cuda:{}".format(self.config.gpu_device[0])


        self.data_loader = SleepDataLoader(config=config)
        self.weights = torch.from_numpy(self.data_loader.weights).float()
        self.loss = nn.CrossEntropyLoss(self.weights.to(self.device))

        enc = self.sleep_load_encoder()
        model_class = globals()[self.config.model_class]
        self.model = model_class(enc)

        print(self.model)
        # self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.config.learning_rate, rho=0.9, eps=1e-06, weight_decay=self.config.weight_decay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2), eps = 1e-08,
                                         weight_decay=self.config.weight_decay)

        if self.config.lr_finder:

            lr_finder = LRFinder(self.model, self.optimizer, self.loss, device=self.device)
            lr_finder.range_test(self.data_loader.train_loader, end_lr=1, num_iter=100)
            _, lr = lr_finder.plot()  # to inspect the loss-learning rate graph
            print("Suggested learning rate is {}".format(lr))
            lr_finder.reset()

        if config.scheduler == "cycle":
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.config.learning_rate, max_lr=self.config.max_lr, cycle_momentum=False)

        elif config.scheduler == "cosanneal":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.max_epoch)

        print("Available cuda devices: {}, current device:{}".format(torch. cuda. device_count(),torch.cuda.current_device()))


        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=[torch.device(i) for i in self.config.gpu_device])
        enc = self.sleep_load_encoder()
        model_class = globals()[self.config.model_class]
        self.best_model = model_class(enc)
        self.best_model = nn.DataParallel(self.best_model, device_ids=[torch.device(i) for i in self.config.gpu_device])

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
        self.current_epoch = 0

        self.train_loss = torch.zeros((self.config.max_epoch, self.config.num_modalities+2)).to(self.device)
        self.val_loss = torch.zeros((self.config.max_epoch, self.config.num_modalities+2)).to(self.device)
        self.test_loss = torch.zeros((self.config.max_epoch, self.config.num_modalities+2)).to(self.device)

        self.train_logs = torch.zeros((self.config.max_epoch, 6)).to(self.device)
        self.best_logs = torch.zeros(5).to(self.device)
        self.best_logs[1] = 5
        self.test_logs = torch.zeros((self.config.max_epoch, 5)).to(self.device)

        self.best_weights = torch.zeros(self.config.num_modalities+1).to(self.device)
        self.gen_weights = torch.zeros((self.config.max_epoch, self.config.num_modalities+1)).to(self.device)

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
            print("Best valid in depoch {0:.2f} with loss {1:.5f} accuracy: {2:.2f}% f1: {3:.4f} kappa: {4:.4f}".format(self.best_logs[0],self.best_logs[1],self.best_logs[2],self.best_logs[3],self.best_logs[4]))

            self.model.load_state_dict(self.best_model.state_dict())
            self.gen_overfit_weigths = self.best_weights

            _, val_acc, val_f1, val_k  = self.sleep_validate()
            print("Validation accuracy: {0:.2f}% and f1: {1:.4f} and k: {2:.4f}".format(val_acc*100, val_f1, val_k))
            print("Weights are {}".format(self.gen_overfit_weigths))

            _, test_acc, test_f1, test_k, test_auc, test_conf = self.sleep_test()

            print("Test accuracy: {0:.2f}% and f1: {1:.4f}".format(test_acc*100,test_f1))
            print("Test kappa: {0:.4f}% and auc: {1:.4f}".format(test_k,test_auc))
            print("Test confusion matrix:")
            print(test_conf)
        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def save_encoder(self, file_name="checkpoint_encoder.pth.tar"):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        if self.config.savetrainedEncoder:
            save_dict = {}
            savior = {}
            savior["encoder_state_dict"] = self.best_model.module.encoder.state_dict()
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

        for self.current_epoch in range(self.current_epoch, self.config.max_epoch):

            for param_group in self.optimizer.param_groups:
                lr =  param_group['lr']
            if self.config.verbose:
                print("We have learning rate: {0:.7f}".format(lr))
            start = time.time()

            self.train_loss[self.current_epoch], train_acc, train_f1, train_k = self.sleep_train_one_epoch()
            self.val_loss[self.current_epoch], val_acc, val_f1, val_k = self.sleep_validate()
            self.train_logs[self.current_epoch] = torch.tensor([train_acc,val_acc,train_f1,val_f1,train_k,val_k],device=self.device)
            not_saved = True

            if self.config.verbose:
                print("Epoch {0:d} Validation loss: {1:.6f}, accuracy: {2:.2f}% f1 :{3:.4f}, k :{4:.4f}  Training loss: {5:.6f}, accuracy: {6:.2f}% f1 :{7:.4f}, k :{8:.4f},".format(self.current_epoch, self.val_loss[self.current_epoch][0], val_acc*100, val_f1, val_k, self.train_loss[self.current_epoch][0], train_acc*100, train_f1, train_k))
            if (self.val_loss[self.current_epoch][0] < self.best_logs[1].item()):
                self.best_logs = torch.tensor([self.current_epoch, self.val_loss[self.current_epoch][0], val_acc, val_f1, val_k],device=self.device)
                self.best_weights = self.gen_overfit_weigths
                self.best_model.load_state_dict(self.model.state_dict())
                print("we have a new best at epoch {0:d} with validation loss: {1:.6f} accuracy: {2:.2f}%, f1: {3:.4f} and k: {4:.4f}".format(self.current_epoch, self.val_loss[self.current_epoch][0], val_acc*100, val_f1, val_k))
                self.sleep_save(self.config.save_dir)
                self.save_encoder(self.config.save_dir_encoder[0])
                not_saved = False
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if (self.current_epoch % self.config.save_every == 0 and not_saved):
                self.sleep_save(self.config.save_dir)
                self.save_encoder(self.config.save_dir_encoder[0])

            if self.config.verbose:
                if self.config.rec_test:
                    self.test_loss[self.current_epoch], test_acc, test_f1, test_k, test_auc, test_conf = self.sleep_test()
                    self.test_logs[self.current_epoch] = torch.tensor([self.current_epoch, self.test_loss[self.current_epoch][0], test_acc, test_f1, test_k], device=self.device)
                    print("Test loss: {0:.6}, accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, auc :{4:.4f}".format(self.test_loss[self.current_epoch][0], test_acc * 100, test_f1, test_k, test_auc))
                print("This epoch took {} seconds with {} no improved epochs".format(time.time() - start,epochs_no_improve) )
            if self.current_epoch > 5 and epochs_no_improve >= self.config.n_epochs_stop:
                early_stop = True
                break
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
        self.sleep_plot_weights()
        return self.best_logs[4]
        # self.save_checkpoint(self.config.save_dir)
        # print("test mse is {}".format(self.test()))

    def compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')

    def sleep_load(self, file_name):
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
        self.train_logs[0:checkpoint["train_logs"].shape[0], :] = checkpoint["train_logs"]
        self.test_logs[0:checkpoint["test_logs"].shape[0], :] = checkpoint["test_logs"]
        self.current_epoch = checkpoint["epoch"]
        self.best_logs = checkpoint["best_logs"]
        self.gen_weights = checkpoint["gen_weights"]
        self.train_loss = checkpoint["train_loss"]
        self.val_loss = checkpoint["val_loss"]
        self.test_loss = checkpoint["test_loss"]
        self.best_weights = checkpoint["best_weights"]
        print("Model has loaded successfully")

    def sleep_load_encoder(self):
        encs = []
        for num_enc in range(self.config.num_modalities):
            enc_class = globals()[self.config.encoder_models[num_enc][0]]
            enc = enc_class(self.config.encoder_models[num_enc][1])
            if self.config.pretrainedEncoder:
                checkpoint = torch.load(self.config.save_dir_encoder[num_enc])
                enc.load_state_dict(checkpoint["encoder_state_dict"])
            encs.append(enc)
        return encs

    def sleep_save(self, file_name="checkpoint.pth.tar"):
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
            savior["seed"] = self.config.seed
            savior["gen_weights"] = self.gen_weights
            savior["train_loss"] = self.train_loss
            savior["val_loss"] = self.val_loss
            savior["test_loss"] = self.test_loss
            savior["best_weights"] = self.best_weights
            save_dict.update(savior)
            try:
                torch.save(save_dict, file_name)
                # if self.config.verbose:
                #     print("Models has saved successfully in {}".format(file_name))
            except:
                raise Exception("Problem in model saving")

    def sleep_train_one_epoch(self):
            """
            One epoch of training
            :return:
            """
            if (self.current_epoch > 1):
                tn = self.calculate_weights(self.data_loader.train_loader)
                vn = self.val_loss[self.current_epoch-1][:self.config.num_modalities+1]

                w = torch.zeros(self.config.num_modalities+1).to(self.device)
                for i in range(self.config.num_modalities+1):
                    gk = self.val_loss[0][i] - vn[i]
                    ok = (self.train_loss[0][i] - self.val_loss[0][i]) - (tn[i] - vn[i])
                    w[i] = gk / ok
                w -= w.min(0, keepdim=True)[0]
                w /= w.max(0, keepdim=True)[0]
                softmax = nn.Softmax(dim=0)
                w = softmax(w)
            else:
                w = torch.ones(self.config.num_modalities+1).to(self.device)
                softmax = nn.Softmax(dim=0)
                w = softmax(w)

            self.gen_weights[self.current_epoch] = w
            self.gen_overfit_weigths = w
            print("Weights are {}".format(self.gen_overfit_weigths.data.cpu().numpy()))

            self.model.train()

            if self.config.freeze_encoders:
                if hasattr(self.model.module,"encoder_eeg"):
                    for p in self.model.module.encoder_eeg.parameters():
                        p.requires_grad = False
                if hasattr(self.model.module,"encoder_stft"):
                    for p in self.model.module.encoder_stft.parameters():
                        p.requires_grad = False
                for i in range(self.config.num_modalities):
                    if hasattr(self.model.module,"enc_{}".format(i)):
                        for p in getattr(self.model.module,"enc_{}".format(i)).parameters():
                            p.requires_grad = False


            batch_loss =  torch.zeros(self.config.num_modalities+2).to(self.device)
            tts, preds = [], []
            pbar = tqdm(enumerate(self.data_loader.train_loader), desc = "Training", leave=False, disable=self.config.tdqm_disable)
            for batch_idx, (data, target, _) in pbar:
                views = [data[i].float().to(self.device) for i in range(len(data))]
                target = target[:,0].to(self.device)
                self.optimizer.zero_grad()
                pred_l = self.model(views)
                loss = [self.loss(p, target) for p in pred_l]
                total_loss = torch.stack([self.gen_overfit_weigths[i]*l for i, l in enumerate(loss)]).sum()
                total_loss.backward()
                #update progress bar
                pbar.set_description("Training batch {0:d}/{1:d} with total loss {2:.5f}".format(batch_idx,len(self.data_loader.train_loader), total_loss.item()))
                pbar.refresh()

                pred = torch.stack([self.gen_overfit_weigths[i]*p for i, p in enumerate(pred_l)]).sum(dim=0)

                l = [total_loss]
                for i in loss: l.append(i)
                batch_loss += torch.stack(l).detach()

                tts.append(target)
                preds.append(pred)
                self.optimizer.step()
            tts = torch.cat(tts).cpu().numpy()
            preds = torch.cat(preds).argmax(axis=1).cpu().numpy()
            return batch_loss / len(self.data_loader.train_loader), np.equal(tts, preds).sum() / len(tts), f1_score(preds, tts), cohen_kappa_score(preds, tts)

    def calculate_weights(self, dataloader):

        self.model.eval()
        with torch.no_grad():
            batch_loss =  torch.zeros(self.config.num_modalities+1).to(self.device)
            pbar = tqdm(enumerate(dataloader), desc="Weights", leave=False,
                        disable=self.config.tdqm_disable)
            for batch_idx, (data, target,_) in pbar:
                views = [data[i].float().to(self.device) for i in range(len(data))]
                target = target[:,0].to(self.device)
                preds = self.model(views)
                loss = [self.loss(p, target) for p in preds]
                batch_loss += torch.stack(loss)
                pbar.set_description("Weights batch {} with loss {}".format(batch_idx, batch_loss))
                pbar.refresh()
        return  batch_loss / len(dataloader)

    def sleep_validate(self):
            """
            One cycle of model validation
            :return:
            """
            self.model.eval()
            batch_loss =  torch.zeros(self.config.num_modalities+2).to(self.device)
            tts, preds, inits = [], [], []
            with torch.no_grad():
                pbar = tqdm(enumerate(self.data_loader.valid_loader), desc="Validation", leave=False,
                            disable=self.config.tdqm_disable)
                for batch_idx, (data, target, _) in pbar:
                    views = [data[i].float().to(self.device) for i in range(len(data))]
                    label = target[:, 0].to(self.device)
                    pred_l = self.model(views)
                    pred = torch.stack([self.gen_overfit_weigths[i] * p for i, p in enumerate(pred_l)]).sum(dim=0)

                    loss = [self.loss(p, label) for p in pred_l]
                    total_loss = self.loss(pred, label)

                    l = [total_loss]
                    for i in loss: l.append(i)
                    batch_loss += torch.stack(l)

                    tts.append(label)
                    inits.append(target[:, 1])
                    preds.append(pred)
                    pbar.set_description("Validation batch {0:d}/{1:d} with loss {2:.5f}".format(batch_idx, len(self.data_loader.valid_loader), total_loss.item()/len(preds)))
                    pbar.refresh()
                tts = torch.cat(tts).cpu().numpy()
                preds = torch.cat(preds).cpu().numpy()
                if self.config.val_postprocessing:
                    preds = self.post_processing(inits, tts, preds)
                preds = preds.argmax(axis=1)
            return batch_loss / len(self.data_loader.valid_loader), np.equal(tts, preds).sum() / len(tts), f1_score(preds, tts), cohen_kappa_score(tts, preds)

    def post_processing(self, inits, tts, preds):

        inits = torch.cat(inits).numpy()
        print("Kappa without postprocessing is k= {0:.4f}".format(cohen_kappa_score(tts, preds.argmax(axis=1))))
        w_idx = int(self.config.post_proc_step / 2 + 1)
        while (w_idx < len(preds) - int(self.config.post_proc_step / 2 + 1)):

            for n_class in range(len(preds[0])):
                preds[w_idx, n_class] = preds[w_idx - int(self.config.post_proc_step / 2):w_idx + int(
                    self.config.post_proc_step / 2), n_class].sum() / self.config.post_proc_step
            if (inits[int(w_idx + int(self.config.post_proc_step / 2))] == 1):
                w_idx += int(self.config.post_proc_step) + 1
            else:
                w_idx += 1
        return preds

    def sleep_test(self):
            """
            One cycle of model validation
            :return:
            """
            print("Test weights are {}".format(self.gen_overfit_weigths))
            self.model.eval()
            batch_loss =  torch.zeros(self.config.num_modalities+2).to(self.device)
            tts, preds, inits = [], [], []
            with torch.no_grad():
                pbar = tqdm(enumerate(self.data_loader.test_loader), desc="Test", leave=False,
                            disable=self.config.tdqm_disable)
                for batch_idx, (data, target, _) in pbar:
                    views = [data[i].float().to(self.device) for i in range(len(data))]
                    label = target[:, 0].to(self.device)
                    pred_l = self.model(views)
                    pred = torch.stack([self.gen_overfit_weigths[i] * p for i, p in enumerate(pred_l)]).sum(dim=0)

                    loss = [self.loss(p, label) for p in pred_l]
                    test_loss = self.loss(pred, label)

                    l = [test_loss]
                    for i in loss: l.append(i)
                    batch_loss += torch.stack(l).detach()

                    tts.append(label)
                    preds.append(pred)
                    inits.append(target[:, 1])
                    pbar.set_description("Test batch {0:d}/{1:d} with total loss {2:.5f}".format(batch_idx, len(self.data_loader.test_loader), test_loss.item()))
                    pbar.refresh()
                tts = torch.cat(tts).cpu().numpy()
                preds = torch.cat(preds).cpu().numpy()
                if self.config.test_postprocessing :
                    preds = self.post_processing(inits, tts, preds)
                preds = preds.argmax(axis=1)
            return batch_loss / len(self.data_loader.test_loader), np.equal(tts, preds).sum() / len(tts), f1_score(preds, tts), cohen_kappa_score(tts, preds), \
                   roc_auc_score(tts, preds), confusion_matrix(tts, preds)

    def sleep_plot_losses(self):
        plt.figure()
        plt.plot(range(self.current_epoch+1), self.train_loss[0:self.current_epoch+1, 0].cpu().numpy(), label="Train")
        plt.plot(range(self.current_epoch+1), self.val_loss[0:self.current_epoch+1, 0].cpu().numpy(), label="Valid")
        plt.plot((self.best_logs[0].item(), self.best_logs[0].item()), (0, self.best_logs[1].item()), linestyle="--",color="y", label="Chosen Point")
        plt.plot((0, self.best_logs[0].item()), (self.best_logs[1].item(), self.best_logs[1].item()), linestyle="--",color="y")

        if self.config.rec_test:
            plt.plot(range(self.current_epoch+1), self.test_loss[0:self.current_epoch+1, 0].cpu().numpy(), label="Test")
            best_test = self.test_logs[:, 1].argmin()
            plt.plot((self.test_logs[best_test, 0].item(), self.test_logs[best_test, 0].item()),
                     (0, self.test_logs[best_test, 1].item()), linestyle="--", color="r", label="Actual Best Loss")
            plt.plot((0, self.test_logs[best_test, 0].item()),
                     (self.test_logs[best_test, 1].item(), self.test_logs[best_test, 1].item()), linestyle="--",
                     color="r")

            best_test = self.test_logs[:, 4].argmax()
            plt.plot((self.test_logs[best_test, 0].item(), self.test_logs[best_test, 0].item()),
                     (0, self.test_logs[best_test, 1].item()), linestyle="--", color="g", label="Actual Best Kappa")
            plt.plot((0, self.test_logs[best_test, 0].item()),
                     (self.test_logs[best_test, 1].item(), self.test_logs[best_test, 1].item()), linestyle="--",
                     color="g")

        plt.xlabel('Epochs')
        plt.ylabel('Loss Values')
        plt.title("Loss")
        plt.legend()
        if self.config.save_plots :
            save_file_name = self.config.save_dir.split(".")[0]+"_loss.png"
            plt.savefig(save_file_name)
        plt.show()

    def sleep_plot_k(self):
        plt.figure()
        plt.plot(range(self.current_epoch+1), self.train_logs[0:self.current_epoch+1, 4].cpu().numpy(), label="Train")
        plt.plot(range(self.current_epoch+1), self.train_logs[0:self.current_epoch+1, 5].cpu().numpy(), label="Valid")
        plt.plot((self.best_logs[0].item(), self.best_logs[0].item()), (0, self.best_logs[4].item()), linestyle="--",
                 color="y", label="Chosen Point")
        plt.plot((0, self.best_logs[0].item()), (self.best_logs[4].item(), self.best_logs[4].item()), linestyle="--",
                 color="y")

        if self.config.rec_test:
            plt.plot(range(self.current_epoch+1), self.test_logs[0:self.current_epoch+1, 4].cpu().numpy(), label="Test")
            best_test = self.test_logs[:, 4].argmax()
            plt.plot((self.test_logs[best_test, 0].item(), self.test_logs[best_test, 0].item()),
                     (0, self.test_logs[best_test, 4].item()), linestyle="--", color="r", label="Actual Best")
            plt.plot((0, self.test_logs[best_test, 0].item()),
                     (self.test_logs[best_test, 4].item(), self.test_logs[best_test, 4].item()), linestyle="--",
                     color="r")

        plt.xlabel('Epochs')
        plt.ylabel("Cohen's Kappa")
        plt.title("Kappa")
        plt.legend()
        if self.config.save_plots :
            save_file_name = self.config.save_dir.split(".")[0]+"_k.png"
            plt.savefig(save_file_name)
        plt.show()

    def sleep_plot_weights(self):
        plt.figure()
        for i in range(len(self.gen_weights[0])):
            label = "Combined" if i == 0 else "View {}".format(i)
            plt.plot(range(self.current_epoch+1), self.gen_weights[0:self.current_epoch+1, i].cpu().numpy(), label=label)
        plt.xlabel('Epochs')
        plt.ylabel("Weights of loss and prediction per view")
        plt.title("Weight Value")
        plt.legend()
        if self.config.save_plots :
            save_file_name = self.config.save_dir.split(".")[0]+"_w.png"
            plt.savefig(save_file_name)
        plt.show()