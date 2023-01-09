import torch
import einops
import copy
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from scipy.stats import entropy
from scipy.special import softmax
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import plotly.graph_objects as go
from os.path import exists

class PatientWise_Analyser_Noisy():
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device

    def validate_specific_patient(self, filename, data_loader, patient_num, device,
                                  std_router=False):

        this_data_loader = copy.deepcopy(data_loader)
        this_data_loader.dataset.choose_specific_patient([patient_num])

        if len(this_data_loader)==0: return {}

        print(patient_num)
        with torch.no_grad():
            keep_views, keep_ids = defaultdict(lambda: []), defaultdict(lambda: [])

            for batch_idx, served_dict in enumerate(this_data_loader):

                served_dict["data"] = {view: served_dict["data"][view].float().to(self.device) for view in served_dict["data"]}

                for v in served_dict["data"]:
                    keep_views[v].append(served_dict["data"][v].cpu())
                    keep_ids[v].append(served_dict["ids"][v])

            keep_views, keep_ids = dict(keep_views), dict(keep_ids)
            single_batch_flag = True if keep_views[list(keep_views.keys())[0]][0].shape[0]==1 else False

            for v in keep_views: keep_views[v] = torch.cat(keep_views[v]).cpu().squeeze()
            for v in keep_ids: keep_ids[v] = torch.cat(keep_ids[v]).cpu().squeeze()
            if single_batch_flag:
                for v in keep_views: keep_views[v] = keep_views[v].unsqueeze(dim=0)
                for v in keep_ids: keep_ids[v] = keep_ids[v].unsqueeze(dim=0)

            if std_router:

                std_router_indications = {}
                for router_key in list(["std_eeg", "std_eog"]):
                    data_key = "time_eeg" if router_key=="std_eeg" else "time_eog"
                    std_router_indications[router_key] = keep_views[data_key].flatten(start_dim=1).std(dim=1).unsqueeze(dim=1).repeat(1, 21)
                    std_router_indications[router_key] = torch.cat([keep_ids["time_eeg"], std_router_indications[router_key].unsqueeze(-1)], dim=-1).flatten(start_dim=0,end_dim=1)

                if exists(filename):

                    file = open(filename,"rb")
                    prev_chosen_indices = pickle.load(file)
                    file.close()
                else:
                    prev_chosen_indices = {}

                prev_chosen_indices.update({patient_num:std_router_indications})
                file = open(filename, "wb")
                pickle.dump(prev_chosen_indices, file)
                file.close()
                return {}

    def find_patient_list(self, data_loader):
        """

        :param data_loader: Pytorch dataloader dataset and with list of dicts {"patient_num": x} in dataset.dataset.cumulatives["files"]
        :return: list of patient_numbers
        """

        patient_list = [data_loader.dataset.cumulatives["files"][i]["patient_num"] for i in data_loader.dataset.cumulatives["files"]]
        patient_list = np.unique(np.array(patient_list))
        print("Total patient number in this dataloader are: {}".format(len(patient_list)))
        print(patient_list)

        return list(patient_list)

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

    def gather_comparisons(self, set, filename, router_models=None, std_router= False):

        if set == "Training": this_dataloader = self.data_loader.train_loader
        elif set == "Validation": this_dataloader = self.data_loader.valid_loader
        elif set == "Test": this_dataloader = self.data_loader.test_loader
        elif set == "Total": this_dataloader = self.data_loader.total_loader
        else: raise ValueError('Set should be one of the "Training", "Validation", "Test" or "Total" ')

        patient_list = self.find_patient_list(data_loader=this_dataloader)

        for i, patient_num in enumerate(patient_list):
            self.validate_specific_patient( filename = filename, data_loader=this_dataloader, device=self.device, patient_num=patient_num, std_router=std_router)
        return {}

    def plot_comparisons(self, models, results, print_entropy=False):

        config_list = list(models.keys())
        comparisons = {"k_comparisons":{}, "f1_comparisons":{}, "acc_comparisons":{}, "entropy_comparisons":{}, "entropy_correct_comparisons":{}, "entropy_wrong_comparisons":{}}
        for conf in config_list:

            comparisons["k_comparisons"][conf] = [results[conf][i]["k"] for i in results[conf].keys()]
            comparisons["f1_comparisons"][conf] = [results[conf][i]["f1"] for i in results[conf].keys()]
            comparisons["acc_comparisons"][conf] = [results[conf][i]["acc"] for i in results[conf].keys()]
            comparisons["entropy_comparisons"][conf] = [results[conf][i]["entropy"] for i in results[conf].keys()]
            comparisons["entropy_correct_comparisons"][conf] = [results[conf][i]["entropy_correct"] for i in results[conf].keys()]
            comparisons["entropy_wrong_comparisons"][conf] = [results[conf][i]["entropy_wrong"] for i in results[conf].keys()]

        config_list = list(models.keys())
        comparisons = {"k_comparisons": {}, "f1_comparisons": {}, "acc_comparisons": {}, "entropy_comparisons": {},
                       "entropy_correct_comparisons": {}, "entropy_wrong_comparisons": {}}
        for conf in config_list:
            comparisons["k_comparisons"][conf] = [results[conf][i]["k"] for i in results[conf].keys()]
            comparisons["f1_comparisons"][conf] = [results[conf][i]["f1"] for i in results[conf].keys()]
            comparisons["acc_comparisons"][conf] = [results[conf][i]["acc"] for i in results[conf].keys()]
            comparisons["entropy_comparisons"][conf] = [results[conf][i]["entropy"] for i in results[conf].keys()]
            comparisons["entropy_correct_comparisons"][conf] = [results[conf][i]["entropy_correct"] for i in
                                                                results[conf].keys()]
            comparisons["entropy_wrong_comparisons"][conf] = [results[conf][i]["entropy_wrong"] for i in
                                                              results[conf].keys()]
        fsize = 20
        ylabel = {"k_comparisons": "Cohens Kappa", "f1_comparisons": "MF1", "acc_comparisons": "Accuracy",
                  "entropy_comparisons": "Entropy",
                  "entropy_correct_comparisons": "Entropy", "entropy_wrong_comparisons": "Entropy"}
        legend_titles = {"eeg": "EEG", "blip_tm_eeg": "CoRe-Sleep EEG", "blip_tm_i": "CoRe-Sleep EEG-EOG", }
        for metric in comparisons:
            if not print_entropy and "entropy" in metric: continue
            print(metric)
            patient_sortargs = np.argsort(comparisons[metric][config_list[0]])
            for conf in config_list:
                comparisons[metric][conf] = [comparisons[metric][conf][i] for i in patient_sortargs]
            models_colors = ["orange", "lightblue", "green", "purple", "red"]
            colors = np.array(comparisons[metric][config_list[0]]) > np.array(comparisons[metric][config_list[1]])
            colors = ["orange" if i else "lightblue" for i in colors]
            plt.figure(figsize=(25, 5))
            ax = plt.subplot(111)
            x = np.linspace(0, len(comparisons[metric][config_list[0]]) - 1, len(comparisons[metric][config_list[0]]))
            plt.xlabel("Patients", fontsize=fsize)
            plt.ylabel(ylabel[metric], fontsize=fsize)
            # plt.title("{} Comparison".format(metric), fontsize = fsize)
            for i, conf in enumerate(config_list):
                plt.plot(x, comparisons[metric][conf], 'o', markersize=11, alpha=0.8, color=models_colors[i],
                         label=legend_titles[conf])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            plt.xticks(fontsize=fsize - 4)
            plt.yticks(fontsize=fsize - 4)
            for i in range(2, 11):
                plt.axhline(i / 10, 0, len(x) - 1, linewidth=0.4, linestyle='--', color="gray", alpha=0.7)
            # for i in range(len(x)):
            #     plt.axvline(x[i], 0, 1, color=colors[i])
            plt.legend(fontsize=fsize, frameon=False, ncol=2, bbox_to_anchor=(0.38, 0.95))
            plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/patient_wise_analysis_test.svg")
            plt.show()

        # colors = np.array(k_comparisons[config_list[0]]) > np.array(k_comparisons[config_list[1]])
        # colors = ["orange" if i else "lightblue" for i in colors]
        # plt.figure(figsize=(25, 5))
        # x = np.linspace(0, len(k_comparisons[config_list[0]]) - 1, len(k_comparisons[config_list[0]]))
        # plt.xlabel("Patient")
        # plt.ylabel("Cohens Kappa")
        # plt.title("K Comparison")
        # plt.plot(x, k_comparisons[config_list[0]], 'o', color='orange', label=config_list[0])
        # plt.plot(x, k_comparisons[config_list[1]], 'o', color='lightblue', label=config_list[1])
        # # for i in range(len(x)):
        # #     plt.axvline(x[i], 0, 1, color=colors[i])
        # plt.legend()
        # plt.show()
        #
        # colors = np.array(acc_comparisons[config_list[0]]) > np.array(acc_comparisons[config_list[1]])
        # colors = ["orange" if i else "lightblue" for i in colors]
        # plt.figure(figsize=(25, 5))
        # x = np.linspace(0, len(acc_comparisons[config_list[0]]) - 1, len(acc_comparisons[config_list[0]]))
        # plt.xlabel("Patient")
        # plt.ylabel("Accuracy")
        # plt.title("Accuracy Comparison")
        # plt.plot(x, acc_comparisons[config_list[0]], 'o', color='orange', label=config_list[0])
        # plt.plot(x, acc_comparisons[config_list[1]], 'o', color='lightblue', label=config_list[1])
        # # for i in range(len(x)):
        # #     plt.axvline(x[i], 0, 1, color=colors[i])
        #
        # plt.legend()
        # plt.show()
        #
        # colors = np.array(entropy_comparisons[config_list[0]]) > np.array(entropy_comparisons[config_list[1]])
        # colors = ["orange" if i else "lightblue" for i in colors]
        # plt.figure(figsize=(25, 5))
        # x = np.linspace(0, len(entropy_comparisons[config_list[0]]) - 1, len(entropy_comparisons[config_list[0]]))
        # plt.xlabel("Patient")
        # plt.ylabel("Entropy")
        # plt.title("Entropy Comparison")
        # plt.plot(x, entropy_comparisons[config_list[0]], 'o', color='orange', label=config_list[0])
        # plt.plot(x, entropy_comparisons[config_list[1]], 'o', color='lightblue', label=config_list[1])
        # # for i in range(len(x)):
        # #     plt.axvline(x[i], 0, 1, color=colors[i])
        # plt.legend()
        # plt.show()
        #
        # colors = np.array(entropy_correct_comparisons[config_list[0]]) > np.array(entropy_correct_comparisons[config_list[1]])
        # colors = ["orange" if i else "lightblue" for i in colors]
        # plt.figure(figsize=(25, 5))
        # x = np.linspace(0, len(entropy_correct_comparisons[config_list[0]]) - 1, len(entropy_correct_comparisons[config_list[0]]))
        # plt.xlabel("Patient")
        # plt.ylabel("Entropy_Correct")
        # plt.title("Entropy_Correct Comparison")
        # plt.plot(x, entropy_correct_comparisons[config_list[0]], 'o', color='orange', label=config_list[0])
        # plt.plot(x, entropy_correct_comparisons[config_list[1]], 'o', color='lightblue', label=config_list[1])
        # # for i in range(len(x)):
        # #     plt.axvline(x[i], 0, 1, color=colors[i])
        # plt.legend()
        # plt.show()
        #
        # colors = np.array(entropy_wrong_comparisons[config_list[0]]) > np.array(entropy_wrong_comparisons[config_list[1]])
        # colors = ["orange" if i else "lightblue" for i in colors]
        # plt.figure(figsize=(25, 5))
        # x = np.linspace(0, len(entropy_wrong_comparisons[config_list[0]]) - 1, len(entropy_wrong_comparisons[config_list[0]]))
        # plt.xlabel("Patient")
        # plt.ylabel("Entropy_Correct")
        # plt.title("Entropy_Correct Comparison")
        # plt.plot(x, entropy_wrong_comparisons[config_list[0]], 'o', color='orange', label=config_list[0])
        # plt.plot(x, entropy_wrong_comparisons[config_list[1]], 'o', color='lightblue', label=config_list[1])
        # # for i in range(len(x)):
        # #     plt.axvline(x[i], 0, 1, color=colors[i])
        # plt.legend()
        # plt.show()

    def load_results(self, filename:str, prev_results:dict={}):
        metrics_file = open(filename, "rb")
        results = pickle.load(metrics_file)
        prev_results.update(results)
        print("Available results:")
        for i in prev_results: print(i)
        return prev_results

    def save_results(self, filename:str, results:dict):

        results = {i: dict(v) for i, v in results.items()} #from defaultdict to dict
        try:
            file = open(filename, "wb")
            pickle.dump(results, file)
            file.close()
            print("Results saved!")
        except:
            print("Error on saving metrics")

    def get_predictions_time_series(self, model, views, inits, skip_modality):
        """
        This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
        :param views: List of tensors, data views/modalities
        :param inits: Tensor indicating with value one, when there incontinuities.
        :return: predictions of the self.model on the batch
        """
        this_inits = inits[list(inits.keys())[0]]
        this_view = views[list(inits.keys())[0]]
        skip_modality_temp = torch.zeros(this_view.shape[0], this_view.shape[1])
        skip_modality_temp[skip_modality["stft_eeg"] == 1] = 1
        skip_modality_temp[skip_modality["stft_eog"] == 1] = 2

        a = ((skip_modality_temp - skip_modality_temp.mean(dim=1).unsqueeze(dim=1).repeat(1, 21)).mean(dim=1) != 0).nonzero(as_tuple=True)[0]
        for batch_idx in a:
            for i in range(len(skip_modality_temp[batch_idx]) - 1):
                if skip_modality_temp[batch_idx][i] != skip_modality_temp[batch_idx][i + 1]:
                    this_inits[batch_idx][i] = 1
                    this_inits[batch_idx][i + 1] = 1

        inits_sum_batch = (this_inits.sum(dim=1) > 1).nonzero(as_tuple=True)[0]
        if len(inits_sum_batch) > 0:
            batch, outer = this_view.shape[0], this_view.shape[1]
            batch_idx_checked = torch.ones(batch, dtype=torch.bool)
            pred ={"preds":{"combined": torch.zeros(batch * outer, 5).to(this_view.device)}}
            for batch_idx in inits_sum_batch:
                ones_idx = (this_inits[batch_idx] > 0).nonzero(as_tuple=True)[0]
                if (ones_idx[0] + 1 == ones_idx[1]  ): #and ones_idx[0]!=0 and ones_idx[1]!= len(inits[idx])
                    if ones_idx[0] == 0:
                        pred_split_0 = model({view: views[view][batch_idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views}, skip_modality={view: skip_modality[view][batch_idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views})
                    else:
                        pred_split_0 = model({view: views[view][batch_idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views}, skip_modality={view: skip_modality[view][batch_idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views})

                    if ones_idx[1] == len(this_inits[batch_idx]):
                        pred_split_1 = model({view: views[view][batch_idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views}, skip_modality={view: skip_modality[view][batch_idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views})
                    else:
                        pred_split_1 = model({view: views[view][batch_idx, ones_idx[1]:].unsqueeze(dim=0) for view in views}, skip_modality={view: skip_modality[view][batch_idx, ones_idx[1]:].unsqueeze(dim=0) for view in views})

                    pred["preds"]["combined"][batch_idx * outer:(batch_idx + 1) * outer] = torch.cat([pred_split_0["preds"]["combined"], pred_split_1["preds"]["combined"]], dim=0)
                else:
                    pred["preds"]["combined"][batch_idx * outer:(batch_idx + 1) * outer] = model({view: views[view][batch_idx].unsqueeze(dim=0) for view in views}, skip_modality={view: skip_modality[view][batch_idx].unsqueeze(dim=0) for view in views})["preds"]["combined"]

                batch_idx_checked[batch_idx] = False
            pred["preds"]["combined"][batch_idx_checked.repeat_interleave(outer)] = model({view: views[view][batch_idx_checked] for view in views}, skip_modality={view: skip_modality[view][batch_idx_checked] for view in views})["preds"]["combined"]

        else:
            pred = model(views, skip_modality=skip_modality)

        return pred

    def plot_spider(self, results):

        categories = ['W', 'N1', 'N2',
                      'N3', 'REM']

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[93.78, 47.57, 89.74, 86.05, 91.38],
            theta=categories,
            fill='toself',
            name='BLIP Limited'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[90.96, 44.9, 86.84, 85.86, 85.05],
            theta=categories,
            fill='toself',
            name='EEG'
        ))

        fig.add_trace(go.Scatterpolar(
            r=[89.03, 23.91, 83.25, 76.23, 87.63],
            theta=categories,
            fill='toself',
            name='EOG'
        ))


        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            legend= dict(
                font = dict(size=30),
                ),
            title = dict(text = "Mode Comparison on BLIP", font = dict(size=30)),

            showlegend=True
        )

        fig.show()



