"""
Mnist Main agent, as mentioned in the tutorial
"""
import numpy as np
from agents.sleep_test.sleep_agent import Sleep_Agent
from graphs.models.custom_unet import *
from utils.misc import print_cuda_statistics
from utils.deterministic_pytorch import deterministic
from datasets.sleepset import SleepDataLoader


class Sleep_Agent_Test(Sleep_Agent):

    def __init__(self, config):
        super().__init__(config)
        deterministic(config.seed)
        self.data_loader = SleepDataLoader(config=config)
        self.device = torch.device(self.config.gpu_device[0])
        self.models = []

        self.weights = torch.from_numpy(self.data_loader.weights).float()
        self.loss = nn.CrossEntropyLoss(self.weights.to(self.device))

        self.vote_weight = np.ones(len(self.config.save_dirs))
        encs = [None for i in range(self.config.num_modalities)]
        for i in range(len(self.config.save_dirs)):
            model_class = globals()[self.config.model_class[i]]
            self.models.append(model_class(encs).to(self.device))
            self.models[i] = nn.DataParallel(self.models[i], device_ids=[torch.device(i) for i in self.config.gpu_device])

        print_cuda_statistics()

    def load_checkpoint(self):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        self.type_models= []
        for idx, dir in enumerate(self.config.save_dirs):
            file_name = self.config.save_root + dir
            print("Loading from file {}".format(file_name))
            checkpoint = torch.load(file_name)
            if "stft_results_" in file_name:
                self.type_models.append(1)
            elif "eeg_results_" in file_name:
                self.type_models.append(0)
            else:
                self.type_models.append(-1)
            self.models[idx].load_state_dict(checkpoint["best_model_state_dict"])
            self.vote_weight[idx] = checkpoint["best_logs"][1]
        print("Models have loaded successfully")

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.load_checkpoint()
            if self.config.test_each:
                test_results = []
                for i in range(len(self.models)):
                    self.model = self.models[i]
                    test_results.append(self.sleep_test())
                    print("Model {0:.1f} has acc: {1:.2f}% and f1: {2:.4f} kappa: {3:.4f}% and auc: {4:.4f}".format(i, test_results[i][1]*100, test_results[i][2], test_results[i][3], test_results[i][4]))
                test_results = np.array(test_results)
                self.test_results = test_results
                print("Mean values are acc: {0:.2f}% and f1: {1:.4f} kappa: {2:.4f}% and auc: {3:.4f}".format(test_results[:,1].mean()*100, test_results[:,2].mean(), test_results[:,3].mean(), test_results[:,4].mean()))
                print("Min kappa {0:.4f} max kappa {1:.4f} and std {2:.4f}".format( test_results[:,3].min(), test_results[:,3].max(),test_results[:,3].std()))
            # self.weight = np.array(test_results)[:,2]
            # print("Weights are {}".format(self.weight))
            test_acc, test_f1, test_k, test_auc, test_conf = self.sleep_test_multi()
            self.k = test_k
            self.ids = test_conf
            print("Ensemble model has:")
            print("Test accuracy: {0:.2f}% and f1: {1:.4f}".format(test_acc * 100, test_f1))
            print("Test kappa: {0:.4f}% and auc: {1:.4f}".format(test_k, test_auc))
            # print("Test confusion matrix:")
            # print(test_conf)

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        print("We are in the final state.")
        if not self.config.test_each:
            return np.zeros(3), self.k, self.ids
        return self.test_results[:,3], self.k, self.ids

    def sleep_test(self):
            """
            One cycle of model validation
            :return:
            """
            self.model.eval()
            test_loss = 0
            from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
            from tqdm import tqdm

            tts = []
            preds = []
            ids = []
            with torch.no_grad():
                for batch_idx, (data, target, id) in enumerate(self.data_loader.test_loader):
                    views = [data[i].float().to(self.device) for i in range(len(data))]
                    target = target.to(self.device)
                    pred = self.model(views)
                    test_loss += self.loss(pred, target).item()
                    tts.append(target)
                    preds.append(pred)
                    ids.append(id)
                tts = torch.cat(tts).cpu().numpy()
                preds = torch.cat(preds).cpu().numpy()
                ids = torch.cat(ids).cpu().numpy()

                for w_idx in range(int(self.config.post_proc_step / 2 + 1), len(preds) - int(self.config.post_proc_step / 2 + 1)):
                    for n_class in range(len(preds[0])):
                        preds[w_idx, n_class] = preds[w_idx - int(self.config.post_proc_step / 2):w_idx + int(self.config.post_proc_step / 2), n_class].sum() / self.config.post_proc_step
                preds = preds.argmax(axis=1)
                test_acc = np.equal(tts, preds).sum() / len(tts)
                test_f1 = f1_score(preds, tts)
                test_k = cohen_kappa_score(tts, preds)
                test_auc = roc_auc_score(tts, preds)
                # test_conf = confusion_matrix(tts, preds)
            return test_loss / len(tts), test_acc, test_f1, test_k, test_auc, {"ids":ids,"found":np.equal(tts, preds)}

    def sleep_test_multi(self):
        for i in range(len(self.models)):
            self.models[i].eval()
        from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
        from tqdm import tqdm

        tts = []
        preds = []
        ids = []
        with torch.no_grad():
            for batch_idx, (data, target, id) in tqdm(enumerate(self.data_loader.test_loader),"Test",leave=False, disable=self.config.tdqm_disable):
                views = [data[i].float().to(self.device) for i in range(len(data))]
                pred = []
                for i in range(len(self.models)):
                    if (self.type_models[i]==0):
                        pred.append(self.models[i]([views[0]])*self.vote_weight[i])
                    elif (self.type_models[i]==1):
                        pred.append(self.models[i]([views[1]])*self.vote_weight[i])
                    else:
                        pred.append(self.models[i](views) * self.vote_weight[i])
                pred = torch.stack(pred, axis=-1)
                pred = pred.mean(axis=-1)
                tts.append(target)
                preds.append(pred)
                ids.append(id)
            tts = torch.cat(tts).cpu().numpy()
            preds = torch.cat(preds).cpu().numpy()
            ids = torch.cat(ids).cpu().numpy()
            import copy
            # m = copy.deepcopy(preds)
            for w_idx in range(int(self.config.post_proc_step/2 +1 ),len(preds)- int(self.config.post_proc_step/2 +1 )):
                for n_class in range(len(preds[0])):
                    preds[w_idx, n_class] = preds[w_idx-int(self.config.post_proc_step/2):w_idx+int(self.config.post_proc_step/2), n_class].sum()/self.config.post_proc_step
            # for w_idx in range(0,len(preds)- int(self.config.post_proc_step),int(self.config.post_proc_step)):
            #     for n_class in range(len(preds[0])):
            #         preds[w_idx:w_idx+int(self.config.post_proc_step), n_class] = preds[w_idx:w_idx+int(self.config.post_proc_step), n_class].sum()/self.config.post_proc_step
            # preds = mc
            preds = preds.argmax(axis=1)
            test_acc = np.equal(tts, preds).sum() / len(tts)
            test_f1 = f1_score(preds, tts)
            test_k = cohen_kappa_score(tts, preds)
            test_auc = roc_auc_score(tts, preds)
            test_conf = confusion_matrix(tts, preds)
        return test_acc, test_f1, test_k, test_auc, {"ids":ids,"found":np.equal(tts, preds)}


