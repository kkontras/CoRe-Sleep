import torch
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from graphs.models.custom_unet import *
from graphs.models.bilstm_att import *
from graphs.models.custom_layers.eeg_encoders import *
import os.path

class Sleep_Agent():
    def __init__(self, config):
        self.config = config
        self.steps_no_improve = 0

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
        self.logs = checkpoint["logs"]
        print("Model has loaded successfully")

    def sleep_load_encoder(self):
        encs = []
        for num_enc in range(len(self.config.encoder_models)):
            enc_class = globals()[self.config.encoder_models[num_enc][0]]
            enc = enc_class(self.config.encoder_models[num_enc][1], self.config.encoder_models[num_enc][2])
            if self.config.pretrainedEncoder[num_enc]:
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
            savior["logs"] = self.logs
            save_dict.update(savior)
            try:
                if self.logs["current_step"] == 0 and os.path.isfile(file_name) :
                    for i in range(10000):
                        new_file_name = file_name.split(".")[-2]+"({})".format(i)+".pth.tar"
                        if not os.path.isfile(new_file_name):
                            break
                    file_name = new_file_name
                    self.config.save_dir = new_file_name
                torch.save(save_dict, file_name)
                if self.config.verbose:
                    print("Models has saved successfully in {}".format(file_name))
            except:
                raise Exception("Problem in model saving")

    def sleep_train_one_epoch(self):
            """
            One epoch of training
            :return:
            """
            self.model.train()
            for enc in range(len(self.config.encoder_models)):
                if self.config.freeze_encoders[i]:
                    if hasattr(self.model.module,"encoder_eeg"):
                        for p in self.model.module.encoder_eeg.parameters():
                            p.requires_grad = False
                    if hasattr(self.model.module,"encoder_stft"):
                        for p in self.model.module.encoder_stft.parameters():
                            p.requires_grad = False
                    if hasattr(self.model.module,"enc_{}".format(i)):
                        for p in getattr(self.model.module,"enc_{}".format(enc)).parameters():
                            p.requires_grad = False

            batch_loss = 0
            tts, preds = [], []
            pbar = tqdm(enumerate(self.data_loader.train_loader), desc = "Training", leave=False, disable=self.config.tdqm_disable)
            for batch_idx, (data, target, _, idxs) in pbar: #tqdm(enumerate(self.data_loader.train_loader), "Training", leave=False, disable=self.config.tdqm_disable):
                views = [data[i].float().to(self.device) for i in range(len(data))]
                target = target.to(self.device).flatten()
                self.optimizer.zero_grad()
                pred = self.model(views)
                loss = self.loss(pred, target)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                loss.backward()
                #update progress bar
                pbar.set_description("Training batch {0:d}/{1:d} with loss {2:.5f}".format(batch_idx,len(self.data_loader.train_loader),loss.item()))
                pbar.refresh()

                batch_loss += loss
                tts.append(target)
                preds.append(pred)
                self.optimizer.step()
                self.scheduler.step()
            # self.model.module.enc_0.dy_conv_0.update_temperature()
            # self.model.module.enc_0.dy_conv_1.update_temperature()
            # self.model.module.enc_0.dy_conv_3.update_temperature()
            tts = torch.cat(tts).cpu().numpy()
            preds = torch.cat(preds).argmax(axis=1).cpu().numpy()
            return batch_loss / len(self.data_loader.train_loader), np.equal(tts, preds).sum() / len(tts), f1_score(preds, tts, average="macro"), cohen_kappa_score(preds, tts, average="macro")

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
            pred = torch.zeros(batch * outer, 5).cuda()
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

    def sleep_train_one_step(self, data, target, inits):


            views = [data[i].float().to(self.device) for i in range(len(data))]
            target = target.to(self.device).flatten()
            self.optimizer.zero_grad()
            pred = self.get_predictions_time_series(views, inits)
            loss = self.loss(pred, target)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()

            return loss, pred

    def monitoring(self, a, b):
        [train_loss, train_acc, train_f1, train_k] = a
        [val_loss, val_acc, val_f1, val_k, val_perclassf1] = b
        self.logs["val_logs"][self.logs["current_step"]] = {"val_loss":val_loss,"val_k":val_k,"val_f1":val_f1,"val_acc":val_acc,"val_perclassf1":val_perclassf1}
        self.logs["train_logs"][self.logs["current_step"]] = {"train_loss":train_loss,"train_k":train_k,"train_f1":train_f1,"train_acc":train_acc,"validate_every":self.config.validate_every,"batch_size":self.config.batch_size}
        early_stop, not_saved, step = False, True, int(self.logs["current_step"] / self.config.validate_every)
        if self.config.verbose:
            print("Epoch {0:d} N: {1:d} Validation loss: {2:.6f}, accuracy: {3:.2f}% f1 :{4:.4f}, k :{5:.4f}  Training loss: {6:.6f}, accuracy: {7:.2f}% f1 :{8:.4f}, k :{9:.4f},".format(
                    self.logs["current_epoch"], self.logs["current_step"] * self.config.batch_size * self.config.seq_legth[0], self.logs["val_logs"][self.logs["current_step"]]["val_loss"], self.logs["val_logs"][self.logs["current_step"]]["val_acc"] * 100,
                    self.logs["val_logs"][self.logs["current_step"]]["val_f1"], self.logs["val_logs"][self.logs["current_step"]]["val_k"], self.logs["train_logs"][self.logs["current_step"]]["train_loss"], self.logs["train_logs"][self.logs["current_step"]]["train_acc"] * 100,
                    self.logs["train_logs"][self.logs["current_step"]]["train_f1"],train_k))
        if (val_loss < self.logs["best_logs"]["val_loss"]):
            self.logs["best_logs"] = {"step":self.logs["current_step"], "val_loss":self.logs["val_logs"][self.logs["current_step"]]["val_loss"], "val_acc":self.logs["val_logs"][self.logs["current_step"]]["val_acc"], "val_f1":self.logs["val_logs"][self.logs["current_step"]]["val_f1"],
                                      "val_k":self.logs["val_logs"][self.logs["current_step"]]["val_k"],  "val_perclassf1":self.logs["val_logs"][self.logs["current_step"]]["val_perclassf1"]}
            print("we have a new best at epoch {0:d} step {1:d} with validation loss: {2:.6f} accuracy: {3:.2f}%, f1: {4:.4f}, k: {5:.4f},  f1_per_class :{6:40}".format(self.logs["current_epoch"], step, val_loss, val_acc * 100, val_f1, val_k, "{}".format(list(val_perclassf1))))
            self.best_model.load_state_dict(self.model.state_dict())
            if self.config.rec_test:
                test_loss, test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens = self.sleep_test()
                print("Test loss: {0:.6}, accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, sens:{4:.4f}, spec:{5:.4f}, f1_per_class :{6:40}".format(test_loss,
                                                                                                         test_acc * 100,
                                                                                                         test_f1,
                                                                                                         test_k, test_spec, test_sens,
                                                                                                         "{}".format(list(test_perclass_f1))))
                self.logs["test_logs"][self.logs["current_step"]] = {"test_loss": test_loss.item(), "test_k": test_k.item(),
                                                                    "test_f1": test_f1.item(), "test_acc": test_acc.item(), "test_spec": test_spec.item(), "test_sens": test_conf,
                                                                    "test_acc": test_conf.item(), "test_auc":test_auc.item(), "test_perclass_f1": list(test_perclass_f1)}
            self.sleep_save(self.config.save_dir)
            self.save_encoder()
            not_saved = False
            self.logs["steps_no_improve"] = 0
        else:
            if self.config.rec_test and self.config.test_on_tops:
                test_loss, test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens = self.sleep_test()
                print(
                    "Test loss: {0:.6}, accuracy: {1:.2f}% f1 :{2:.4f}, k :{3:.4f}, sens:{4:.4f}, spec:{5:.4f}, f1_per_class :{6:40}".format(test_loss,
                                                                                                         test_acc * 100,
                                                                                                         test_f1,
                                                                                                         test_k, test_spec, test_sens,
                                                                                                         "{}".format(list(test_perclass_f1))))
                self.logs["test_logs"][self.logs["current_step"]] = {"test_loss": test_loss, "test_k": test_k,
                                                                    "test_f1": test_f1, "test_acc": test_acc, "test_spec": test_spec, "test_sens": test_conf,
                                                                    "test_acc": test_conf, "test_auc":test_auc, "test_perclass_f1": test_perclass_f1}
            self.logs["steps_no_improve"] += 1
            print("Current steps with no improvement {}".format(self.logs["steps_no_improve"]))
        if (self.logs["current_step"] % self.config.save_every == 0 and not_saved):
            self.sleep_save(self.config.save_dir)
            self.save_encoder()

        if self.config.verbose:
            print("This epoch took {} seconds".format(time.time() - self.start))
        if self.logs["current_epoch"] > self.config.n_epochs_stop_after and self.logs["steps_no_improve"] >= self.config.n_steps_stop:
            print('Early stopping!')
            early_stop = True

        return early_stop

    def sleep_train_step(self):
        self.model.train()

        tts, preds, batch_loss, datapoints_sum, early_stop = [], [], 0, 0, False
        self.start = time.time()
        for self.logs["current_epoch"] in range(self.logs["current_epoch"], self.config.max_epoch):
            pbar = tqdm(enumerate(self.data_loader.train_loader), desc="Training", leave=None, disable=self.config.tdqm_disable, position=0)
            for batch_idx, (data, target, inits, idxs) in pbar:
                if len(target.shape)>2:
                    target = target.flatten()
                self.model.train()
                loss, pred = self.sleep_train_one_step(data, target, inits)
                batch_loss += loss.item()
                datapoints_sum+=1
                tts.append(target)
                preds.append(pred)
                pbar.set_description("Training batch {0:d}/{1:d} with loss {2:.5f}".format(batch_idx, len(self.data_loader.train_loader), batch_loss/datapoints_sum))
                pbar.refresh()
                if self.logs["current_step"] % self.config.validate_every == 0 and batch_idx!=0:
                    # print("We are in validation")

                    del data, target
                    tts = torch.cat(tts).cpu().numpy().flatten()
                    preds = torch.cat(preds).argmax(axis=1).cpu().numpy()
                    train_loss, train_acc, train_f1, train_k = batch_loss / self.config.validate_every, np.equal(tts, preds).sum() / len(tts), f1_score(preds,tts,average="macro"), cohen_kappa_score(preds, tts)
                    batch_loss, datapoints_sum, tts, preds = 0, 0, [], []
                    val_loss, val_acc, val_f1, val_k, val_perclassf1 = self.sleep_validate()
                    # self.model.train()
                    early_stop = self.monitoring([train_loss, train_acc, train_f1, train_k],[val_loss, val_acc, val_f1, val_k, val_perclassf1])
                    if early_stop: break
                    self.start = time.time()

                self.logs["current_step"] += 1
            if early_stop: break

        return batch_loss / len(self.data_loader.train_loader), np.equal(tts, preds).sum() / len(tts), f1_score(preds,tts, average="macro"), cohen_kappa_score(preds, tts)

    def sleep_validate(self):
            """
            One cycle of model validation
            :return:
            """
            self.model.eval()
            valid_loss = 0
            tts, preds, inits = [], [], []
            # hidden = None
            with torch.no_grad():
                pbar = tqdm(enumerate(self.data_loader.valid_loader), desc="Validation", leave=False,
                            disable=True, position=1)
                for batch_idx, (data, target, init, _) in pbar:

                    views = [data[i].float().to(self.device) for i in range(len(data))]
                    label = target.to(self.device).flatten()
                    pred = self.get_predictions_time_series(views, init)

                    # pred = self.model(views, init)
                    # hidden = torch.stack(list(hidden), dim=0)
                    valid_loss += self.loss(pred, label).item()
                    tts.append(label)
                    preds.append(pred)
                    inits.append(init.flatten())
                    pbar.set_description("Validation batch {0:d}/{1:d} with total loss {2:.5f}".format(batch_idx, len(self.data_loader.valid_loader), valid_loss/len(preds)))
                    pbar.refresh()
                tts = torch.cat(tts).cpu().numpy()
                preds = torch.cat(preds).cpu().numpy()
                if self.config.val_postprocessing :
                    inits = torch.cat(inits).numpy()
                    w_idx = int(self.config.post_proc_step / 2 + 1)
                    while (w_idx < len(preds) - int(self.config.post_proc_step / 2 + 1)):

                        for n_class in range(len(preds[0])):
                            preds[w_idx, n_class] = preds[w_idx - int(self.config.post_proc_step / 2):w_idx + int(
                                self.config.post_proc_step / 2), n_class].sum() / self.config.post_proc_step
                        if (inits[int(w_idx + int(self.config.post_proc_step / 2))] == 1):
                            w_idx += int(self.config.post_proc_step) + 1
                        else:
                            w_idx += 1
                preds = preds.argmax(axis=1)
            return valid_loss / len(self.data_loader.valid_loader), np.equal(tts, preds).sum() / len(tts), f1_score(preds, tts, average="macro"), cohen_kappa_score(
                tts, preds), f1_score(preds, tts, average=None)

    def sleep_test(self):
            """
            One cycle of model validation
            :return:
            """
            self.model.eval()
            test_loss = 0

            tts, preds, inits = [], [], []
            with torch.no_grad():
                pbar = tqdm(enumerate(self.data_loader.test_loader), desc="Test", leave=False,
                            disable=True, position=2)
                for batch_idx, (data, target, init, _) in pbar:
                    views = [data[i].float().to(self.device) for i in range(len(data))]
                    label = target.to(self.device).flatten()
                    pred = self.get_predictions_time_series(views, init)

                    test_loss += self.loss(pred, label).item()
                    tts.append(label)
                    preds.append(pred)
                    inits.append(init.flatten())
                    pbar.set_description("Test batch {0:d}/{1:d} with total loss {2:.5f}".format(batch_idx, len(self.data_loader.test_loader), test_loss))
                    pbar.refresh()
                tts = torch.cat(tts).cpu().numpy()
                preds = torch.cat(preds).cpu().numpy()
                if self.config.test_postprocessing :
                    inits = torch.cat(inits).numpy()
                    print("Test kappa without postprocessing is k= {0:.4f}".format(cohen_kappa_score(tts, preds.argmax(axis=1))))
                    w_idx = int(self.config.post_proc_step / 2 + 1)
                    while (w_idx < len(preds) - int(self.config.post_proc_step / 2 + 1)):

                        for n_class in range(len(preds[0])):
                            preds[w_idx, n_class] = preds[w_idx - int(self.config.post_proc_step / 2):w_idx + int(
                                self.config.post_proc_step / 2), n_class].sum() / self.config.post_proc_step
                        if (inits[int(w_idx + int(self.config.post_proc_step / 2 ))] == 1 ):
                            w_idx += int(self.config.post_proc_step) + 1
                        else:
                            w_idx+=1

                multiclass = False
                if preds.shape[1]>2:
                    multiclass = True
                preds = preds.argmax(axis=1)
                test_acc = np.equal(tts, preds).sum() / len(tts)
                test_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average = "macro")
                test_perclass_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average = None)
                test_k = cohen_kappa_score(tts, preds)
                test_auc = roc_auc_score(tts, preds) if not multiclass else 0
                test_conf = confusion_matrix(tts, preds)
                tp, fp, tn, fn = self._perf_measure(tts, preds)
                test_spec = tn / (tn+fp)
                test_sens = tp / (tp+fn)
            return test_loss / len(self.data_loader.test_loader), test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens

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
        train_loss = np.array([self.logs["train_self.logs"][i]["train_loss"] for i in self.logs["train_self.logs"]])
        val_loss = np.array([self.logs["val_self.logs"][i]["val_loss"] for i in self.logs["val_self.logs"]])
        steps = np.array([i / self.logs["train_self.logs"][i]["validate_every"] for i in self.logs["train_self.logs"]]) - 1

        plt.figure()
        plt.plot(steps, train_loss, label="Train")
        plt.plot(steps, val_loss, label="Valid")

        best_step = self.logs["best_self.logs"]["step"] / self.logs["train_self.logs"][self.logs["best_self.logs"]["step"]][
            "validate_every"] - 1
        best_loss = self.logs["best_self.logs"]["val_loss"]

        plt.plot((best_step, best_step), (0, best_loss), linestyle="--", color="y", label="Chosen Point")
        plt.plot((0, best_step), (best_loss, best_loss), linestyle="--", color="y")

        if self.config.rec_test:
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

        if self.config.rec_test:
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

        if self.config.rec_test:
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