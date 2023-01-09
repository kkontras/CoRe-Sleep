import torch
import tqdm
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

def sleep_load(obj, file_name):
    """
    Latest checkpoint loader
    :param file_name: name of the checkpoint file
    :return:
    """
    print("Loading from file {}".format(file_name))
    checkpoint = torch.load(file_name)
    obj.model.load_state_dict(checkpoint["model_state_dict"])
    obj.best_model.load_state_dict(checkpoint["best_model_state_dict"])
    obj.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    obj.train_logs[0:checkpoint["train_logs"].shape[0], :] = checkpoint["train_logs"]
    obj.test_logs[0:checkpoint["test_logs"].shape[0], :] = checkpoint["test_logs"]
    obj.current_epoch = checkpoint["epoch"]
    obj.best_logs = checkpoint["best_logs"]
    print("Model has loaded successfully")

def sleep_save(obj, file_name="checkpoint.pth.tar"):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        save_dict = {}
        savior = {}
        savior["model_state_dict"] = obj.model.state_dict()
        savior["best_model_state_dict"] = obj.best_model.state_dict()
        savior["optimizer_state_dict"] = obj.optimizer.state_dict()
        savior["train_logs"] = obj.train_logs
        savior["test_logs"] = obj.test_logs
        savior["epoch"] = obj.current_epoch
        savior["best_logs"] = obj.best_logs
        savior["seed"] = obj.config.seed
        save_dict.update(savior)
        try:
            torch.save(save_dict, file_name)
            print("Models has saved successfully in {}".format(file_name))
        except:
            raise Exception("Problem in model saving")

def sleep_train_one_epoch(obj):
        """
        One epoch of training
        :return:
        """
        obj.model.train()
        batch_loss = 0
        tts, preds = [], []
        for batch_idx, (data, target, _) in tqdm(enumerate(obj.data_loader.train_loader), "Training", leave=False, disable=obj.config.tdqm):  # enumerate(obj.data_loader.train_loader):
            views = [data[i].float().to(obj.device) for i in range(len(data))]
            target = target.to(obj.device)
            obj.optimizer.zero_grad()
            pred = obj.model(views)
            loss = obj.loss(pred, target)
            loss.backward()
            batch_loss += loss
            tts.append(target)
            preds.append(pred)
            obj.optimizer.step()
            obj.scheduler.step()
        tts = torch.cat(tts).cpu().numpy()
        preds = torch.cat(preds).argmax(axis=1).cpu().numpy()
        return batch_loss / len(tts), np.equal(tts, preds).sum() / len(tts), f1_score(preds, tts), cohen_kappa_score(preds, tts)

def sleep_validate(obj):
        """
        One cycle of model validation
        :return:
        """
        obj.model.eval()
        valid_loss = 0
        tts, preds = [], []
        with torch.no_grad():
            for batch_idx, (data, target, _) in tqdm(enumerate(obj.data_loader.valid_loader), "Validation",
                                                     leave=False,
                                                     disable=obj.config.tdqm):  # enumerate(obj.data_loader.valid_loader):
                views = [data[i].float().to(obj.device) for i in range(len(data))]
                target = target.to(obj.device)
                pred = obj.model(views)
                valid_loss += obj.loss(pred, target).item()
                tts.append(target)
                preds.append(pred)
            tts = torch.cat(tts).cpu().numpy()
            preds = torch.cat(preds).cpu().numpy()
            for w_idx in range(int(obj.config.post_proc_step / 2 + 1),
                               len(preds) - int(obj.config.post_proc_step / 2 + 1)):
                for n_class in range(len(preds[0])):
                    preds[w_idx, n_class] = preds[w_idx - int(obj.config.post_proc_step / 2):w_idx + int(
                        obj.config.post_proc_step / 2), n_class].sum() / obj.config.post_proc_step
            preds = preds.argmax(axis=1)
        return valid_loss / len(tts), np.equal(tts, preds).sum() / len(tts), f1_score(preds, tts), cohen_kappa_score(
            tts, preds)

def sleep_test(obj):
        """
        One cycle of model validation
        :return:
        """
        obj.model.eval()
        test_loss = 0

        tts = []
        preds = []
        with torch.no_grad():
            for batch_idx, (data, target, _) in tqdm(enumerate(obj.data_loader.test_loader), "Test", leave=False,
                                                     disable=obj.config.tdqm):
                views = [data[i].float().to(obj.device) for i in range(len(data))]
                target = target.to(obj.device)
                pred = obj.model(views)
                test_loss += obj.loss(pred, target).item()
                tts.append(target)
                preds.append(pred)
            tts = torch.cat(tts).cpu().numpy()
            preds = torch.cat(preds).cpu().numpy()
            for w_idx in range(int(obj.config.post_proc_step / 2 + 1),
                               len(preds) - int(obj.config.post_proc_step / 2 + 1)):
                for n_class in range(len(preds[0])):
                    preds[w_idx, n_class] = preds[w_idx - int(obj.config.post_proc_step / 2):w_idx + int(
                        obj.config.post_proc_step / 2), n_class].sum() / obj.config.post_proc_step
            preds = preds.argmax(axis=1)
            test_acc = np.equal(tts, preds).sum() / len(tts)
            test_f1 = f1_score(preds, tts)
            test_k = cohen_kappa_score(tts, preds)
            test_auc = roc_auc_score(tts, preds)
            test_conf = confusion_matrix(tts, preds)
        return test_loss / len(tts), test_acc, test_f1, test_k, test_auc, test_conf

def sleep_plot_losses(obj):
    plt.figure()
    plt.plot(range(obj.current_epoch), obj.train_logs[0:obj.current_epoch, 1].cpu().numpy(), label="Train")
    plt.plot(range(obj.current_epoch), obj.train_logs[0:obj.current_epoch, 0].cpu().numpy(), label="Valid")
    plt.plot(range(obj.current_epoch), obj.test_logs[0:obj.current_epoch, 1].cpu().numpy(), label="Test")
    plt.plot((obj.best_logs[0].item(), obj.best_logs[0].item()), (0, obj.best_logs[1].item()), linestyle="--",
             color="y", label="Chosen Point")
    plt.plot((0, obj.best_logs[0].item()), (obj.best_logs[1].item(), obj.best_logs[1].item()), linestyle="--",
             color="y")

    if obj.config.rec_test:
        best_test = obj.test_logs[:, 1].argmin()
        plt.plot((obj.test_logs[best_test, 0].item(), obj.test_logs[best_test, 0].item()),
                 (0, obj.test_logs[best_test, 1].item()), linestyle="--", color="r", label="Actual Best Loss")
        plt.plot((0, obj.test_logs[best_test, 0].item()),
                 (obj.test_logs[best_test, 1].item(), obj.test_logs[best_test, 1].item()), linestyle="--",
                 color="r")

        best_test = obj.test_logs[:, 4].argmax()
        plt.plot((obj.test_logs[best_test, 0].item(), obj.test_logs[best_test, 0].item()),
                 (0, obj.test_logs[best_test, 1].item()), linestyle="--", color="g", label="Actual Best Kappa")
        plt.plot((0, obj.test_logs[best_test, 0].item()),
                 (obj.test_logs[best_test, 1].item(), obj.test_logs[best_test, 1].item()), linestyle="--",
                 color="g")

    # plt.ylim(bottom= 0.0005)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Values')
    plt.title("Loss")
    plt.legend()
    plt.show()

def sleep_plot_k(obj):
    plt.figure()
    plt.plot(range(obj.current_epoch), obj.train_logs[0:obj.current_epoch, 6].cpu().numpy(), label="Train")
    plt.plot(range(obj.current_epoch), obj.train_logs[0:obj.current_epoch, 7].cpu().numpy(), label="Valid")
    plt.plot(range(obj.current_epoch), obj.test_logs[0:obj.current_epoch, 4].cpu().numpy(), label="Test")
    plt.plot((obj.best_logs[0].item(), obj.best_logs[0].item()), (0, obj.best_logs[4].item()), linestyle="--",
             color="y", label="Chosen Point")
    plt.plot((0, obj.best_logs[0].item()), (obj.best_logs[4].item(), obj.best_logs[4].item()), linestyle="--",
             color="y")

    if obj.config.rec_test:
        best_test = obj.test_logs[:, 4].argmax()
        plt.plot((obj.test_logs[best_test, 0].item(), obj.test_logs[best_test, 0].item()),
                 (0, obj.test_logs[best_test, 4].item()), linestyle="--", color="r", label="Actual Best")
        plt.plot((0, obj.test_logs[best_test, 0].item()),
                 (obj.test_logs[best_test, 4].item(), obj.test_logs[best_test, 4].item()), linestyle="--",
                 color="r")

    # plt.ylim(bottom= 0.3)
    plt.xlabel('Epochs')
    plt.ylabel("Cohen's Kappa")
    plt.title("Kappa")
    plt.legend()
    plt.show()

def sleep_plot_eeg(obj, data):
    time = np.arange(0, 30 - 1 / 900, 30 / 900)
    plt.figure("EEG Window")
    data = data.squeeze()
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(time, data[0][i])
    plt.show()