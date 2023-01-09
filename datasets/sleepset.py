import einops
from torch.utils.data import DataLoader, Dataset, Subset
# from torchvision.transforms import ToTensor
import numpy as np
import csv
import torch
import os
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
from PIL import Image
import copy
from scipy import signal as sg
import random
import pickle
import zarr
import h5py
from tqdm import tqdm
from scipy.io import loadmat
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import scipy
import sys
import psutil
from sklearn.model_selection import KFold
# import matlab.engine
from collections import defaultdict

# eng = matlab.engine.start_matlab()

def _init_fn(worker_id):
    """
    This function is fed into the dataloaders to have deterministic shuffle.
    :param worker_id:
    :return:
    """
    np.random.seed(15 + worker_id)

class Transform_Images():
    def __init__(self, enh):
        super()
        self.sos = {}
        self.coef = {}
        self.nyq = 0.5 * 30

        for k in enh.keys():
            if k.isnumeric() and enh[str(k)]["method"] == "Extract_Freq" :
                if enh[str(k)]["filt_type"] == "butter":
                    self.sos[str(k)] = sg.butter(enh[str(k)]["order"], [enh[str(k)]["filt_cutoffs"][0] / self.nyq, enh[str(k)]["filt_cutoffs"][1] / self.nyq],
                              btype='band', output="sos")
                elif enh[str(k)]["filt_type"] == "fir":
                    self.coef[str(k)] = sg.firwin(fs=30, numtaps=enh[str(k)]["order"], cutoff=[enh[str(k)]["filt_cutoffs"][0], enh[str(k)]["filt_cutoffs"][1]], window="hamming", pass_zero=False)

    def Gaussian_Noise_Add(self, image, enh, num):
        img_shape = image.shape
        aug = torch.normal(mean = enh["mean"], std = enh["std"], size=tuple(img_shape))
        return image + aug

    def Extract_Freq(self, image, enh, num):

        # "2": {"method": "Extract_Freq", "filt_type": "fir", "order": 50, "filt_cutoffs": [2, 3]},
        # "3": {"method": "Extract_Freq", "filt_type": "fir", "order": 50, "filt_cutoffs": [13, 14.8]}},

        if enh["filt_type"] == "butter":
            image = torch.from_numpy(np.array([sg.sosfiltfilt(self.sos[str(num)], signal) for signal in image[0,:,:]])).unsqueeze(dim=0)
        elif enh["filt_type"] == "fir":
            image = torch.from_numpy(np.array([sg.filtfilt(self.coef[str(num)], [1], signal) for signal in image[0,:,:]])).unsqueeze(dim=0)
        return image

    def Zero_Mask(self, image, enh, num):
        max_ch_change = random.randint(0, enh["max_ch"])
        for i in range(max_ch_change):
            ch_to_change = random.randint(0, image.shape[-2]-1)
            zero_region = random.randint(0, enh["max_zero_region"])
            zero_start = random.randint(0, image.shape[-1] - zero_region)
            image[:,ch_to_change,zero_start:zero_start+zero_region ] = torch.zeros(zero_region)
        return image

    def Noise_Mask(self, image, enh, num):
        max_ch_change = random.randint(0, enh["max_ch"])
        for i in range(max_ch_change):
            ch_to_change = random.randint(0, image.shape[-2]-1)
            zero_region = random.randint(0, enh["max_zero_region"])
            zero_start = random.randint(0, image.shape[-1] - zero_region)
            image[:,ch_to_change,zero_start:zero_start+zero_region ] = torch.normal(mean = enh["mean"], std = enh["std"], size=(1, zero_region))[0]
        return image

    def Amp_Change(self, image, enh, num):
        max_ch_change = random.randint(0, enh["max_ch"])
        for i in range(max_ch_change):
            ch_to_change = random.randint(0, image.shape[-2]-1)
            zero_region = random.randint(0, enh["max_zero_region"])
            zero_start = random.randint(0, image.shape[-1] - zero_region)
            image[:,ch_to_change,zero_start:zero_start+zero_region ] = torch.ones(zero_region) * torch.rand(1)
        return image

class Sleep_Dataset_mat_huy(Dataset):

    def __init__(self, config, views, set_name, data_augmentation= {}):
        super()
        self.dataset = views
        self.config = config
        self.set_name = set_name
        self.data_augmentation = data_augmentation

        self._init_attributes()

        if self.filter_windows["use_type"]:
            self._find_list_of_patients()
            self.broken_mod_dict = self._get_broken_mod(filename="/users/sista/kkontras/Documents/Sleep_Project/experiments/baddiff_chosen_shhs_total.pkl")

        self._get_cumulatives()
        self._get_len()

        self._get_teacher_preds()

    def _init_attributes(self):
        self.num_views = len(self.dataset)
        self.seq_views = self.config.dataset.seq_views
        self.keep_view = self.config.dataset.keep_view
        self.inner_overlap = self.config.dataset.inner_overlap
        self.normalize = self.config.normalization.use
        self.outer_seq_length = self.config.dataset.seq_length[0]
        self.inner_seq_length = self.config.dataset.seq_length[1]

        self.clean_train = self.config.model.args.clean_train if "clean_train" in self.config.model.args else False

        self.filter_windows = self.config.dataset.filter_windows[self.set_name] \
            if "filter_windows" in self.config.dataset and self.set_name in self.config.dataset.filter_windows\
            else {"use_type": False, "skip_skips":True}

        print(self.filter_windows)
        self.mask_channel = torch.rand([1, 1, 1, 129, 29]).float().numpy()
        self.mask_label = torch.rand([1, 5]).long()
        self.mask_init = torch.rand([1]).long()

        self.data_augm_times = len(self.data_augmentation.keys()) - 2
        self.augmentation_type = self.data_augmentation['type'] if 'type' in self.data_augmentation.keys() else 'same'
        self.augmentation_rate = self.data_augmentation['rate'] if 'rate' in self.data_augmentation.keys() else 0
        self.aug = self.data_augmentation
        self.tf = Transform_Images(self.data_augmentation)

    def _get_teacher_preds(self):
        if "teacher_pred" in self.config.model.args:
            with open('./teacher_predictions.pickle', 'rb') as handle:
                self.teachers_pred = pickle.load(handle)

            #A couple were missing, quick solution!
            self.teachers_pred["/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_EDFX/Version_1/train/patient_0077/file_00/n0077_f00_eeg_stft.hdf5"][2762] = np.array([9.9999619e-01, 8.1474741e-07, 2.0287409e-06, 1.2491529e-07, 8.6466622e-07])
            self.teachers_pred["/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_EDFX/Version_1/train/patient_0077/file_00/n0077_f00_eeg_stft.hdf5"][2763] = np.array([9.9999619e-01, 8.1474741e-07, 2.0287409e-06, 1.2491529e-07, 8.6466622e-07])

        # metrics = self.get_normalized_values()

    def _find_list_of_patients(self):
        """
        Method to find the list of patient_numbers included in the dataset/subset
        :return: self.patient_list -> list of patient numbers
        """
        self.patient_list = []
        for view in self.dataset:
            for file_idx in range(len(self.dataset[view]["dataset"])):
                file = self.dataset[view]["dataset"][file_idx]
                patient_num = int(file["filename"].split("/")[-1][1:5])
                self.patient_list.append(patient_num)
        self.patient_list = np.array(self.patient_list)
        self.patient_list = np.unique(self.patient_list)

    def _get_len(self):
        # self.dataset_true_length = int(np.array([int(g) for g in self.dataset[1]]).sum()/self.outer_seq_length)
        self.dataset_true_length = int(self.cumulatives["lengths"][-1]/self.outer_seq_length)
        # self.dataset_true_length = int(np.array([int(int(g)/self.outer_seq_length) for g in self.dataset[1]]).sum())

        if self.augmentation_type == "mul":
            self.dataset_aug_length = int(self.dataset_true_length * self.data_augm_times + 1)
        elif self.augmentation_type == "same":
            self.dataset_aug_length = self.dataset_true_length

    def _subsample_patients(self, std_per_indices):

        combined_set = random.sample(list(self.patient_list), self.filter_windows["subsets"]["combined"])
        skip_patient_ids = {patient: torch.cat([torch.zeros(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1), torch.zeros(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1)], dim=1) for patient in combined_set}
        not_chosen_patients = [patient for patient in self.patient_list if patient not in combined_set]

        eeg_set = random.sample(not_chosen_patients, self.filter_windows["subsets"]["eeg"])
        a = {patient: torch.cat([torch.zeros(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1), torch.ones(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1)], dim=1) for patient in eeg_set}
        skip_patient_ids.update(a)
        not_chosen_patients = [patient for patient in self.patient_list if patient not in combined_set]

        eog_set = random.sample(not_chosen_patients, self.filter_windows["subsets"]["eog"])
        a = {patient: torch.cat([torch.ones(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1), torch.zeros(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1)], dim=1) for patient in eog_set}
        skip_patient_ids.update(a)

        self.subsampled_patients = {"combined": combined_set,"eeg":eeg_set, "eog": eog_set, "all":combined_set + eeg_set + eog_set}
        print("We subsample patients with {} both mods, {} only EEG and {} only EOG and len_skipped is {}.".format(len(combined_set), len(eeg_set), len(eog_set), len(skip_patient_ids)))

        return skip_patient_ids

    def _get_broken_mod(self, filename):

        # metrics_file = open(filename,"rb")
        # bad_perf_patients = pickle.load(metrics_file)
        # patient_nums_unique, patient_nums_counts = np.unique(bad_perf_patients[name][:, :, 0].flatten().numpy(), return_counts=True)
        # skip_patient_ids_i = {int(patient_num): bad_perf_patients[name][bad_perf_patients[name][:, :, 0] == patient_num][:, 1:] for patient_num in patient_nums_unique}



        filename = "/users/sista/kkontras/Documents/Sleep_Project/experiments/noisy_patients_trial2_pp.pkl"
        file = open(filename, "rb")
        std_per_indices = pickle.load(file)
        file.close()

        if self.filter_windows["use_type"] == "subsample":
           skip_patient_ids = self._subsample_patients(std_per_indices)
           return skip_patient_ids

        threshold = self.filter_windows["std_threshold"]
        perc_threshold = self.filter_windows["perc_threshold"]

        mod_diff = {i: (std_per_indices[i]["std_eeg"][:, 2] - std_per_indices[i]["std_eog"][:, 2]).numpy() for i in list(std_per_indices.keys()) if i in self.patient_list}
        perc_t = {i: (np.abs(mod_diff[i]) > threshold).sum()/len(mod_diff[i]) for i in list(mod_diff.keys())}
        patients_chosen = np.array([i for i in perc_t if perc_t[i] > perc_threshold])

        print("Patients with broken modalities are {}".format(len(patients_chosen)))
        skip_patient_ids = {}
        for i in patients_chosen:
            skip_patient_ids[i] = []
            for j in range(len(mod_diff[i])):
                if mod_diff[i][j]>0 and mod_diff[i][j] > threshold:
                    skip_mod = [1,0]
                elif mod_diff[i][j]<0 and mod_diff[i][j] < -threshold:
                    skip_mod = [0,1]
                else:
                    skip_mod = [0,0]
                skip_patient_ids[i].append(skip_mod)
            skip_patient_ids[i] = torch.from_numpy(np.array(skip_patient_ids[i]))

        #Bar plot of percentage difference of EEG-EOG thresholded.
        # perc_t_bar = np.array([perc_t[i] for i in perc_t])
        # plt.bar(np.arange(0,len(perc_t_bar)), np.sort(perc_t_bar))
        # plt.title("Percentile of std difference |EEG-EOG| > {} in recording".format(threshold))
        # plt.ylabel("Percentile")
        # plt.xlabel("Patient")
        # plt.xticks([0,499,999,1999, 2999, 3999, 4999, len(perc_t35)], [1, 500, 1000, 2000, 3000, 4000, 5000, len(perc_t35)+1])
        # plt.show()

        return skip_patient_ids

    def _get_cumulatives(self):
        view = list(self.dataset.keys())[0]
        self.cumulatives = {"lengths":[0], "files":{}}
        for file_idx in range(len(self.dataset[view]["dataset"])):
            file = self.dataset[view]["dataset"][file_idx]
            patient_num = int(file["filename"].split("/")[-1][1:5])
            file_len = int(file["len_windows"])
            if "discard_30_mins" in self.config.dataset and self.config.dataset.discard_30_mins:
                raise NotImplementedError()
            #     self.cumulatives["lengths"].append(file_len-120 +self.cumulatives["lengths"][-1])
            #     start_idx = 60
            #     end_idx = file_len - 60
            #     self.cumulatives["files"]["{}-{}".format(self.cumulatives["lengths"][-2],self.cumulatives["lengths"][-1])]={
            #         "patient_num": patient_num,
            #         "data_idx": {"start_idx":start_idx, "end_idx": end_idx},
            #         "dataset":{view: self.dataset[view]["dataset"][file_idx] for view in self.dataset}
            #     }
            # elif self.filter_windows["use"]:
            if self.filter_windows["use_type"] == "include_only_skipped":
                self._single_patient_cumulative_includeskipped(patient_num, file_len, file_idx)
            elif self.filter_windows["use_type"] == "subsample":
                if not self.filter_windows["whole_patient"]: raise Warning("Whole patients is not true which might lead to different sampling rather the desired one. Possibly disclude patients with both modalities available")
                self._single_patient_cumulative_includeskipped(patient_num, file_len, file_idx)
            elif self.filter_windows["use_type"] == "exclude_only_skipped":
                self._single_patient_cumulative_excludeskipped(patient_num, file_len, file_idx)
            else:
                self._single_patient_cumulative_full(patient_num, file_len, file_idx)

    def _single_patient_cumulative_includeskipped(self, patient_num, file_len, file_idx):
        if patient_num in self.broken_mod_dict:

            if self.filter_windows["whole_patient"]:
                self._single_patient_cumulative_full(patient_num, file_len, file_idx)
                return 0

            this_broken_patient = copy.deepcopy(self.broken_mod_dict[patient_num])
            this_broken_patient[:,1:2] *= 2
            skip_labels, skip_labels_lengths = torch.unique_consecutive(this_broken_patient.sum(dim=1), return_counts=True)
            count, consecutives = 0, []
            for i in range(len(skip_labels)):
                end = count + skip_labels_lengths[i]
                consecutives.append({"start": count, "end": end, "skip_label":skip_labels[i],
                                     "skip_views": {
                    "stft_eeg": self.broken_mod_dict[patient_num][count:count + skip_labels_lengths[i],:1].squeeze(),
                    "stft_eog": self.broken_mod_dict[patient_num][count:count + skip_labels_lengths[i],1:].squeeze()}})
                count = count + skip_labels_lengths[i]


            # consecutives = []
            # consecutives_total = 0
            # consecutive_start = self.broken_mod_dict[patient_num][0, 0]
            # for id in range(len(self.broken_mod_dict[patient_num][:, 0]) - 1):
            #     if (self.broken_mod_dict[patient_num][id + 1, 0] - self.broken_mod_dict[patient_num][id, 0] != 1) or (
            #             self.broken_mod_dict[patient_num][id + 1, 1] != self.broken_mod_dict[patient_num][id, 1]):
            #         consecutive_end = self.broken_mod_dict[patient_num][id, 0]
            #         consecutives.append({"start": int(consecutive_start), "end": int(consecutive_end),
            #                              "skip_label": int(self.broken_mod_dict[patient_num][id, 1])})
            #         consecutives_total += int(consecutive_end) - int(consecutive_start)
            #         consecutive_start = self.broken_mod_dict[patient_num][id + 1, 0]
            #     elif id + 2 == len(self.broken_mod_dict[patient_num][:, 0]):
            #         consecutive_end = self.broken_mod_dict[patient_num][id + 1, 0]
            #         consecutives.append({"start": int(consecutive_start), "end": int(consecutive_end),
            #                              "skip_label": int(self.broken_mod_dict[patient_num][id, 1])})
            #         consecutives_total += int(consecutive_end) - int(consecutive_start)

            for cons in consecutives:
                if cons["skip_label"] == 3 or cons["skip_label"] == 0: continue
                self.cumulatives["lengths"].append(cons["end"] - cons["start"] + self.cumulatives["lengths"][-1])


                self.cumulatives["files"]["{}-{}".format(self.cumulatives["lengths"][-2], self.cumulatives["lengths"][-1])] = \
                    {
                        "patient_num": patient_num,
                        "data_idx": {"start_idx": cons["start"], "end_idx": cons["end"]},
                        "dataset": {view: self.dataset[view]["dataset"][file_idx] for view in
                                    self.dataset},
                        "skip_views": cons["skip_views"]
                    }

    def _single_patient_cumulative_excludeskipped(self, patient_num, file_len, file_idx):
        if patient_num in self.broken_mod_dict:
            consecutives = []
            consecutives_total = 0
            consecutive_start = self.broken_mod_dict[patient_num][0, 0]
            for id in range(len(self.broken_mod_dict[patient_num][:, 0]) - 1):
                if (self.broken_mod_dict[patient_num][id + 1, 0] - self.broken_mod_dict[patient_num][id, 0] != 1) or (
                        self.broken_mod_dict[patient_num][id + 1, 1] != self.broken_mod_dict[patient_num][id, 1]):
                    consecutive_end = self.broken_mod_dict[patient_num][id, 0]
                    consecutives.append({"start": int(consecutive_start), "end": int(consecutive_end),
                                         "skip_label": int(self.broken_mod_dict[patient_num][id, 1])})
                    consecutives_total += int(consecutive_end) - int(consecutive_start)
                    consecutive_start = self.broken_mod_dict[patient_num][id + 1, 0]
                elif id + 2 == len(self.broken_mod_dict[patient_num][:, 0]):
                    consecutive_end = self.broken_mod_dict[patient_num][id + 1, 0]
                    consecutives.append({"start": int(consecutive_start), "end": int(consecutive_end),
                                         "skip_label": int(self.broken_mod_dict[patient_num][id, 1])})
                    consecutives_total += int(consecutive_end) - int(consecutive_start)

        else:
            consecutives = [{"start": 0, "end": file_len,  "skip_label":0}]

        if self.filter_windows["whole_patient"]:
            for cons in consecutives:
                if cons["skip_label"]!=0:
                    # print("Patient {} contains some skipped so we skip him".format(patient_num))
                    return "Patient {} contains some skipped so we skip him".format(patient_num)

        for cons in consecutives:
            if cons["skip_label"]==0:
                self.cumulatives["lengths"].append(cons["end"] - cons["start"] + self.cumulatives["lengths"][-1])

                self.cumulatives["files"][
                    "{}-{}".format(self.cumulatives["lengths"][-2],
                                   self.cumulatives["lengths"][-1])] = {
                "patient_num": patient_num,
                "data_idx": {"start_idx": cons["start"], "end_idx": cons["end"]},
                "dataset": {view: self.dataset[view]["dataset"][file_idx] for view in
                            self.dataset}
            }

    def _single_patient_cumulative_subsample(self, patient_num, file_len, file_idx):
        self.cumulatives["lengths"].append(file_len + self.cumulatives["lengths"][-1])
        start_idx = 0
        end_idx = file_len
        self.cumulatives["files"]["{}-{}".format(self.cumulatives["lengths"][-2], self.cumulatives["lengths"][-1])] = {
            "patient_num": patient_num,
            "data_idx": {"start_idx": start_idx, "end_idx": end_idx},
            "dataset": {view: self.dataset[view]["dataset"][file_idx] for view in self.dataset}
        }

    def _single_patient_cumulative_full(self, patient_num, file_len, file_idx):
        start_idx = 0
        end_idx = file_len

        if hasattr(self, "broken_mod_dict") and patient_num in self.broken_mod_dict:
            this_broken_patient = copy.deepcopy(self.broken_mod_dict[patient_num])
            this_broken_patient[:,1:2] *= 2
            skip_labels, skip_labels_lengths = torch.unique_consecutive(this_broken_patient.sum(dim=1), return_counts=True)
            count, consecutives = 0, []
            for i in range(len(skip_labels)):
                end = count + skip_labels_lengths[i]
                consecutives.append({"start": count, "end": end, "skip_label":skip_labels[i],
                                     "skip_views": {
                    "stft_eeg": self.broken_mod_dict[patient_num][count:count + skip_labels_lengths[i],:1].squeeze(),
                    "stft_eog": self.broken_mod_dict[patient_num][count:count + skip_labels_lengths[i],1:].squeeze()}})
                count = count + skip_labels_lengths[i]
        else:
            consecutives = [
                {"start": start_idx,
                 "end": end_idx,
                 "skip_views":{
                    "stft_eeg": torch.zeros(end_idx-start_idx),
                    "stft_eog": torch.zeros(end_idx-start_idx)}
                 }
            ]

        for cons in consecutives:
            self.cumulatives["lengths"].append(cons["end"]-cons["start"] + self.cumulatives["lengths"][-1])

            self.cumulatives["files"]["{}-{}".format(self.cumulatives["lengths"][-2], self.cumulatives["lengths"][-1])] = {
                "patient_num": patient_num,
                "data_idx": {"start_idx": cons["start"], "end_idx": cons["end"]},
                "dataset": {view: self.dataset[view]["dataset"][file_idx] for view in self.dataset},
                "skip_views": cons["skip_views"]
            }
        # if hasattr(self, "broken_mod_dict") and patient_num in self.broken_mod_dict:
        #     if type(self.broken_mod_dict[patient_num]) == str:
        #         skip_str_to_Tensor = {"both": torch.Tensor([0]), "eog": torch.Tensor([1]), "eeg": torch.Tensor([2])}
        #         skip_views = {view: skip_str_to_Tensor[self.broken_mod_dict[patient_num]].repeat(end_idx-start_idx) for view in self.dataset}
        #     elif type(self.broken_mod_dict[patient_num]) == list:
        #         skip_views = {view: self.broken_mod_dict[patient_num][view][start_idx:end_idx] for view in self.dataset}
        #
        #     self.cumulatives["files"]["{}-{}".format(self.cumulatives["lengths"][-2], self.cumulatives["lengths"][-1])]["skip_views"] = skip_views

    def get_normalized_values(self):
        metrics = {"mean":np.zeros([129,]),"mean_sq":np.zeros([129]),"sum":0,"count_labels":np.zeros([5])}
        for filename in self.dataset[0]:
            with h5py.File(filename, 'r') as f:

                if metrics["sum"] == 0:
                    metrics["mean"] += np.array(f["X2"]).mean(axis=1).mean(axis=1)
                    X2_squared = np.square(np.array(f["X2"]))
                    meanXsquared_i = X2_squared.mean(axis=1).mean(axis=1)
                    metrics["mean_sq"] += meanXsquared_i
                    metrics["sum"] += f["X2"].shape[-1] * f["X2"].shape[-2]
                else:
                    meanX_i = np.array(f["X2"]).mean(axis=1).mean(axis=1)
                    X2_squared = np.square(np.array(f["X2"]))
                    meanXsquared_i = X2_squared.mean(axis=1).mean(axis=1)
                    Ni = f["X2"].shape[-1] * f["X2"].shape[-2]
                    metrics["mean"] = (metrics["mean"] * metrics["sum"] + meanX_i * Ni) / (metrics["sum"] + Ni)
                    metrics["mean_sq"] = (metrics["mean_sq"] * metrics["sum"] + meanXsquared_i * Ni) / (metrics["sum"] + Ni)
                    metrics["sum"] += Ni
                for l in  f["label"]:
                    metrics["count_labels"][int(l)] += 1
        varX = -np.multiply(metrics["mean"], metrics["mean"]) + metrics["mean_sq"]
        metrics["std"] = np.sqrt(varX*metrics["sum"]/(metrics["sum"]-1))
        return metrics

    def loadnnorm_mat(self, file, file_idx, mod, num_channels):

        file_name = file["dataset"][mod]["filename"]
        data_idx = file["data_pos"]
        data_num = file["data_num"]
        end_file = file["end_file"]
        start_file = file["start_file"]
        end_idx = data_num + data_idx
        # skip_view = file["skip_views"][mod][data_idx:end_idx]
        if "skip_skips" in self.filter_windows and self.filter_windows["skip_skips"]:
            skip_view = torch.empty(0)
        else:
            skip_view = file["skip_views"][mod]

        # skip_view_bool = []
        # for i in range(data_idx, end_idx):
        #     skip_view_bool.append(1 if i in skip_view else 0)
        # skip_view_bool = torch.Tensor(skip_view_bool)

        # index_skip = skip_view_bool==0

        # f = loadmat(file[0])
        f = h5py.File(file_name, 'r', swmr=True)
        label = f["labels"][data_idx:end_idx]
        init = torch.zeros(len(label))

        if "stft" in mod:

            signal = []
            for key in range(num_channels):
                if "X2_ch_{}".format(key) in f.keys():
                    this_channel = np.expand_dims(f["X2_ch_{}".format(key)][data_idx:end_idx], axis=1)
                    if len(this_channel.shape) != 4:
                        raise NotImplementedError()

                        # skip_channel[key] = 1
                        # signal.append(self.mask_channel.repeat(end_idx-data_idx, axis=0))
                        # continue
                    signal.append(this_channel)
                else:
                    raise NotImplementedError()
                    # signal.append(self.mask_channel.repeat(end_idx-data_idx, axis=0))
            signal = np.concatenate(signal, axis=1)

            # signal = np.expand_dims(signal, axis=1)
            if self.normalize and hasattr(self, "mean") and hasattr(self, "std"):
                signal = einops.rearrange(signal, "inner channels freq time -> inner time channels freq")
                if self.config.normalization.metrics_type == "per_recording":
                    if type(self.mean[file_idx][mod]) is np.ndarray:
                        tmp_v = self.mean[file_idx][mod]
                        self.mean[file_idx][mod] = {}
                        self.mean[file_idx][mod]["concat"] = tmp_v
                        tmp_v = self.std[file_idx][mod]
                        self.std[file_idx][mod] = {}
                        self.std[file_idx][mod]["concat"] = tmp_v
                    else:
                        if "concat" not in self.mean[file_idx][mod]:
                            self.mean[file_idx][mod]["concat"] = np.concatenate(
                                [np.expand_dims(self.mean[file_idx][mod]["ch_{}".format(key)], axis=0) for key in
                                 range(num_channels)], axis=0).squeeze()

                        if "concat" not in self.std[file_idx][mod]:
                            self.std[file_idx][mod]["concat"] = np.concatenate(
                                [np.expand_dims(self.std[mod]["ch_{}".format(key)], axis=0) for key in
                                 range(num_channels)], axis=0).squeeze()

                    signal = (signal - self.mean[file_idx][mod]["concat"]) / self.std[file_idx][mod]["concat"]
                else:
                    if type(self.mean[mod]) is np.ndarray:
                        tmp_v = self.mean[mod]
                        self.mean[mod] = {}
                        self.mean[mod]["concat"] = tmp_v
                        tmp_v = self.std[mod]
                        self.std[mod] = {}
                        self.std[mod]["concat"] = tmp_v
                    else:
                        if "concat" not in self.mean[mod]:
                            self.mean[mod]["concat"] = np.concatenate(
                                [np.expand_dims(self.mean[mod]["ch_{}".format(key)], axis=0) for key in
                                 range(num_channels)], axis=0).squeeze()

                        if "concat" not in self.std[mod]:
                            self.std[mod]["concat"] = np.concatenate(
                                [np.expand_dims(self.std[mod]["ch_{}".format(key)], axis=0) for key in
                                 range(num_channels)], axis=0).squeeze()

                    signal = (signal - self.mean[mod]["concat"]) / self.std[mod]["concat"]
                signal = einops.rearrange(signal, "inner time channels freq -> inner channels freq time")
            else:
                signal = einops.rearrange(signal, "freq channels time inner -> inner channels freq time")

        elif "time" in mod:
            signal = []
            for key in range(num_channels):
                if "X1_ch_{}".format(key) in f.keys():
                    this_channel = np.expand_dims(np.expand_dims(f["X1_ch_{}".format(key)][data_idx:end_idx], axis=1),
                                                  axis=1)
                    signal.append(this_channel)
                else:
                    raise NotImplementedError()
            signal = np.concatenate(signal, axis=2)

            if self.normalize and hasattr(self, "mean") and hasattr(self, "std"):
                signal = einops.rearrange(signal, "time inner -> inner time")
                # signal = (signal - self.mean[mod]) / self.std[mod]
            else:
                signal = einops.rearrange(signal, "time inner -> inner time")

        if data_idx == start_file and end_file > data_idx and len(init) > 1:
            init[0] = 1
        elif end_idx == end_file:
            init[-1] = 1

        img = torch.from_numpy(signal).unsqueeze(dim=2)
        label = torch.from_numpy(label) - 1
        ids = [{"patient_num": file_idx, "ids": i} for i in range(data_idx, end_idx)]

        return {"data": img, "label": label, "init": init, "ids": ids, "skip_view": skip_view}


    def loadnnorm_mat_huy(self, file, file_idx, mod, num_channels):

        file_name = file["dataset"][mod]["filename"]
        data_idx = file["data_pos"]
        data_num = file["data_num"]
        end_file = file["end_file"]
        start_file = file["start_file"]
        end_idx = data_num + data_idx

        if "skip_skips" not in self.filter_windows or self.filter_windows["skip_skips"]:
            skip_view = torch.empty(0)
        else:
            skip_view = file["skip_views"][mod]

        # skip_view_bool = []
        # for i in range(data_idx, end_idx):
        #     skip_view_bool.append(1 if i in skip_view else 0)
        # skip_view_bool = torch.Tensor(skip_view_bool)

        # index_skip = skip_view_bool==0

        # f = loadmat(file[0])
        f = h5py.File(file_name, 'r', swmr=True)
        label = f["label"][0,data_idx:end_idx]
        init = torch.zeros(len(label))

        if "stft" in mod:

            # print(f.keys())
            # #For NCH
            # signal = np.concatenate([np.expand_dims(f[key][data_idx:end_idx,:,:], axis=1) for key in f.keys() if "X2" in key], axis=1)
            # print(signal.shape)
            # if self.normalize and hasattr(self,"mean") and hasattr(self,"std"):
            #     signal = einops.rearrange(signal, "time channels freq inner -> inner time channels freq")
            #     signal = (signal - self.mean[mod]) / self.std[mod]
            #     signal = einops.rearrange(signal, "inner time channels freq -> inner channels freq time")
            # else:
            #     signal = einops.rearrange(signal, "time channels freq inner -> inner channels freq time")
            # print(signal.shape)


            #For SHHS
            signal = f["X2"][:,:,data_idx:end_idx]
            signal = np.expand_dims(signal, axis=1)
            if self.normalize and hasattr(self,"mean") and hasattr(self,"std"):
                signal = einops.rearrange(signal, "freq channels time inner -> inner time channels freq")
                if self.config.normalization.metrics_type == "per_recording":
                    if type(self.mean[file_idx][mod]) is np.ndarray:
                        tmp_v = self.mean[file_idx][mod]
                        self.mean[file_idx][mod] = {}
                        self.mean[file_idx][mod]["concat"] = tmp_v
                        tmp_v = self.std[file_idx][mod]
                        self.std[file_idx][mod] = {}
                        self.std[file_idx][mod]["concat"] = tmp_v
                    else:
                        if "concat" not in self.mean[file_idx][mod]:
                            self.mean[file_idx][mod]["concat"] = np.concatenate(
                                [np.expand_dims(self.mean[file_idx][mod]["ch_{}".format(key)], axis=0) for key in
                                 range(num_channels)], axis=0).squeeze()

                        if "concat" not in self.std[file_idx][mod]:
                            self.std[file_idx][mod]["concat"] = np.concatenate(
                                [np.expand_dims(self.std[mod]["ch_{}".format(key)], axis=0) for key in
                                 range(num_channels)], axis=0).squeeze()

                    signal = (signal - self.mean[file_idx][mod]["concat"]) / self.std[file_idx][mod]["concat"]
                else:
                    if type(self.mean[mod]) is np.ndarray:
                        tmp_v = self.mean[mod]
                        self.mean[mod] = {}
                        self.mean[mod]["concat"] = tmp_v
                        tmp_v = self.std[mod]
                        self.std[mod] = {}
                        self.std[mod]["concat"] = tmp_v
                    else:
                        if "concat" not in self.mean[mod]:
                            self.mean[mod]["concat"] = np.concatenate(
                                [np.expand_dims(self.mean[mod]["ch_{}".format(key)], axis=0) for key in
                                 range(num_channels)], axis=0).squeeze()

                        if "concat" not in self.std[mod]:
                            self.std[mod]["concat"] = np.concatenate(
                                [np.expand_dims(self.std[mod]["ch_{}".format(key)], axis=0) for key in
                                 range(num_channels)], axis=0).squeeze()

                    signal = (signal - self.mean[mod]["concat"]) / self.std[mod]["concat"]
                signal = einops.rearrange(signal, "inner time channels freq -> inner channels freq time")
            else:
                signal = einops.rearrange(signal, "freq channels time inner -> inner channels freq time")

        elif "time" in mod:
            signal = f["X1"][:,data_idx:end_idx]
            if self.normalize and hasattr(self,"mean") and hasattr(self,"std"):
                signal = einops.rearrange(signal, "time inner -> inner time")
                # signal = (signal - self.mean[mod]) / self.std[mod]
            else:
                signal = einops.rearrange(signal, "time inner -> inner time")

        # if data_idx == start_file and end_file>data_idx and  len(init) >1:
        if data_idx == start_file and end_file>data_idx and  len(init) >0:
            init[0] = 1
        elif end_idx == end_file:
            init[-1] = 1

        img = torch.from_numpy(signal).unsqueeze(dim=2)
        label = torch.from_numpy(label) -1
        ids = [{ "patient_num":file_idx,"ids": i} for i in range(data_idx, end_idx)]

        # print(skip_view)
        # if len(skip_view) == 0:
        #     print("hello")
        return {"data": img, "label": label, "init": init, "ids":ids, "skip_view":skip_view}

    def set_mean_std(self, mean, std):
        self.mean = mean
        self.std = std

    def check_index_file(self, file_i, previous_output) -> dict:
        if "index" not in previous_output: raise ValueError("Missing attribute 'index' in previous_output of self.chech_file in dataset")
        if "remaining_epochs" not in previous_output: raise ValueError("Missing attribute 'remaining_epochs' in previous_output of self.chech_file in dataset")
        index = previous_output["index"]
        remaining_epochs = previous_output["remaining_epochs"]
        patient_num = -1
        if self.cumulatives["lengths"][file_i + 1] > index and self.cumulatives["lengths"][file_i] <= index:
            cumul_file = self.cumulatives["files"]["{}-{}".format(self.cumulatives["lengths"][file_i], self.cumulatives["lengths"][file_i + 1])]
            data_idx = index - self.cumulatives["lengths"][file_i] + cumul_file["data_idx"]["start_idx"]
            new_remaining_epochs = max(remaining_epochs - (cumul_file["data_idx"]["end_idx"] - data_idx), 0)
            data_num = remaining_epochs - new_remaining_epochs

            previous_output["index"] += data_num
            previous_output["remaining_epochs"] = new_remaining_epochs
            new_output = {"data_pos": data_idx, "data_num": data_num,
                          "end_file": cumul_file["data_idx"]["end_idx"],
                          "start_file": cumul_file["data_idx"]["start_idx"],
                          "remaining_epochs": previous_output["remaining_epochs"],
                          "dataset": cumul_file["dataset"]
                          }
            patient_num = cumul_file["patient_num"]
            # if patient_num==1254:
            #     print("here")
            if "skip_views" not in cumul_file:
                new_output["skip_views"] = {view: torch.empty(0) for view in self.dataset}
            elif type(cumul_file["skip_views"][list(cumul_file["skip_views"].keys())[0]]) == torch.Tensor and ("skip_skips" in self.filter_windows and not self.filter_windows["skip_skips"]):
                new_output["skip_views"] = {view: cumul_file["skip_views"][view][data_idx-cumul_file['data_idx']['start_idx']:data_idx+data_num-cumul_file['data_idx']['start_idx']] for view in self.dataset}


            # #TODO: Move these in cumul file, so that it includes the corresponding skip view for each file/timespan
            # skip_views = {view: [] for view in self.dataset}
            # if hasattr(self, "broken_mod_dict") and int(cumul_file["patient_num"]) in self.broken_mod_dict:
            #     for i in range(data_idx, data_idx+data_num):
            #         exist_element = (self.broken_mod_dict[int(cumul_file["patient_num"])][:, 0] == i).nonzero(as_tuple=True)[0]
            #         if len(exist_element)>0:
            #             if self.broken_mod_dict[int(cumul_file["patient_num"])][exist_element[0], 1] == 1:
            #                 skip_mod = "eeg"
            #             elif self.broken_mod_dict[int(cumul_file["patient_num"])][exist_element[0], 1] == 2:
            #                 skip_mod = "eog"
            #             elif self.broken_mod_dict[int(cumul_file["patient_num"])][exist_element[0], 1] == 0:
            #                 skip_mod = "none"
            #             elif self.broken_mod_dict[int(cumul_file["patient_num"])][exist_element[0], 1] == 3:
            #                 skip_mod = "both"
            #             for view in self.dataset:
            #                 if skip_mod in view or skip_mod=="both":skip_views[view].append(i)
            # new_output["skip_views"] = skip_views

            if cumul_file["patient_num"] in previous_output:
                previous_output[cumul_file["patient_num"]].append(new_output)
            else:
                previous_output.update({cumul_file["patient_num"]: [new_output]})

        return previous_output, patient_num

    def __getitem__(self, index):

        index = index * self.outer_seq_length # cast it from batch to epoch space number.

        file_output = {"remaining_epochs": self.outer_seq_length, "index": index}
        #find the files that correspond to the given index!
        for file_i in range(len(self.cumulatives["lengths"]) - 1):
            #check files and cumulative to find which one we have to open and which parts of the file to use.
            file_output, last_pat_num = self.check_index_file(file_i=file_i, previous_output=file_output)
            if (last_pat_num in file_output) and file_output[last_pat_num][-1]["remaining_epochs"]==0: break
        #remove useless_info
        file_output.pop("index")
        file_output.pop("remaining_epochs")

        # print(file_output)

        #file_output is organised as
        # {#num_of_file:{
              # "dataset":{"view_name": list_of_files, "second_view_name":list_of_files} # each file is a dict with "filename" and "len_windows" how many windows the file contains
              # "data_pos": int, # which position in the dataset we should get the data from
              # "data_num": int, # how many should we get from this position
              # "remaining": int, # how many remain to fill the outer_sequence
              # },
              # {"dataset":list_of_files},
              # }

        loadNnorm_func = self.loadnnorm_mat_huy if self.config.dataset.huy_data else self.loadnnorm_mat

        nested_dict = lambda: defaultdict(nested_dict)
        total_output = nested_dict()
        for view in file_output[list(file_output.keys())[0]][0]["dataset"]: # stft_eeg, stft_eog, time_eeg, time_eog etc.
            loaded_output = defaultdict(lambda: [])
            for file_idx in file_output:
                for num_seqs in range(len(file_output[file_idx])):
                    #TODO: Add this num channel to file_output somewhere.
                    num_channels = 1 #self.config.dataset.data_view_dir[int(seq_files / step_file)]["num_ch"]
                    loaded_data = loadNnorm_func( file=file_output[file_idx][num_seqs],
                                                  file_idx=file_idx,
                                                  mod=view, num_channels=num_channels)
                    for i in loaded_data:
                        loaded_output[i].append(loaded_data[i])

            for out_i in dict(loaded_output):
                if out_i == "ids":
                    loaded_output["ids"] = [item for sublist in loaded_output["ids"] for item in sublist]
                    patient_nums = torch.cat([torch.Tensor([int(id["patient_num"])]) for id in loaded_output["ids"]])
                    ids = torch.cat([torch.Tensor([int(id["ids"])]) for id in loaded_output["ids"]])
                    total_ids = torch.cat([patient_nums.unsqueeze(dim=1), ids.unsqueeze(dim=1)],dim=1)

                    total_output[out_i].update({view:total_ids})
                    continue
                total_output[out_i].update({view: torch.cat(loaded_output[out_i])})

            total_output["label"][view] = total_output["label"][view].float() if "softlabels" in self.config.dataset and self.config.dataset.softlabels else total_output["label"][view].long()
            if len(total_output["label"][view].shape) > 1:
                    total_output["label"][view] = total_output["label"][view].argmax(dim=1)

            if "random_shuffle_data" in self.config.dataset and self.config.dataset.random_shuffle_data:
                raise NotImplementedError
                perms = torch.randperm(views[0].shape[1])
                views = [view[:, perms] for view in views]
                target = einops.rearrange(target, "(batch seq) -> batch seq", batch=views[0].shape[0],
                                          seq=views[0].shape[1])[:, perms].flatten()
                init = init[:, perms]

            if "random_shuffle_data_batch" in self.config.dataset and self.config.dataset.random_shuffle_data_batch:
                raise NotImplementedError

                perms = torch.randperm(views[0].shape[0] * views[0].shape[1])
                d_shape = views[0].shape
                views = [einops.rearrange(einops.rearrange(view, "batch seq b c d -> (batch seq) b c d")[perms],
                                          "(batch seq) b c d -> batch seq b c d", batch=d_shape[0],
                                          seq=d_shape[1])
                         for view in views]
                target = target.flatten()[perms]
                init = einops.rearrange(einops.rearrange(init, "batch seq -> (batch seq)")[perms],
                                        "(batch seq) -> batch seq", batch=d_shape[0], seq=d_shape[1])

        if "skip_skips" in self.filter_windows and self.filter_windows["skip_skips"]:
            total_output["skip_view"] = torch.empty(0)
        #Reshape img to create the inner windows
        if self.inner_seq_length != 0:
            raise NotImplementedError
            for i, img in enumerate(images):
                if self.keep_view[i] == 1:
                    img_shape = list(img.shape)
                    assert img_shape[-1] % self.inner_seq_length ==0, "Quants of time in each view/modality must be divisable by the inner sequence length"
                    dim = 1 if self.outer_seq_length > 1 else 0
                    start_index = 0
                    windows = []

                    window_samples = int(img.shape[-1] / self.inner_seq_length)
                    inner_oversample = int(window_samples*self.inner_overlap[i])
                    assert inner_oversample != 0, "Overlapping in the inner sequence length is not possible"
                    #TODO: For some reason we dont take the last 1.5 secs or the 30 samples. Investigate that.
                    while (start_index + window_samples < img.shape[-1]+1):
                        if len(img.shape) == 3:
                            current_window = img[:, :, start_index:start_index + window_samples]
                        elif len(img.shape) == 4:
                            current_window = img[:, :, :, start_index:start_index + window_samples]
                        elif len(img.shape) == 5:
                            current_window = img[:, :, :, :, start_index:start_index + window_samples]
                        start_index = int(start_index + inner_oversample)
                        windows.append(current_window.unsqueeze(dim=dim))
                    windows = torch.cat(windows, dim = dim)
                    if self.seq_views[i]:
                        output.append(windows)
                    else:
                        output[i] = windows

        return total_output

    def __len__(self):
        return self.dataset_aug_length

    def preload_data(self):
        data_len = self.__len__()
        g_output, g_labels, g_init, g_ids = [], [], [], []
        pbar = tqdm(range(data_len), desc="Pre-loading validation data", leave=False)
        pre_loaded_idx = data_len
        for i in pbar:
            output, label, init, ids = self.__getitem__(index=i)
            g_output.append(output)
            g_labels.append(label)
            g_init.append(init)
            g_ids.append(ids)
            output_size = sys.getsizeof(g_output)
            labels_size = sys.getsizeof(g_labels)
            init_size = sys.getsizeof(g_init)
            ids_size = sys.getsizeof(g_ids)

            total_size = psutil.virtual_memory().percent

            # total_size = output_size + labels_size + init_size + ids_size


            pbar.set_description("Pre-loading validation data {0:d} / {1:d}  RAM is {2:.1f}%".format(i, data_len, total_size))
            pbar.refresh()
            if total_size > self.config.byte_limits:
                pre_loaded_idx = i
                break

        total_size = output_size + labels_size + init_size + ids_size

        print("Cashed are {0:.0f} Gb and {1:.3f} Mb".format(total_size//(10**9),(total_size%(10**9))/(10**6)))

        return g_output, g_labels, g_init, g_ids, pre_loaded_idx

    def choose_specific_patient(self, patient_nums, include_chosen=True):

        for view in self.dataset:
            new_view_dataset = []
            for file in self.dataset[view]["dataset"]:
                if include_chosen:
                    if int(file["filename"].split("/")[-1][1:5]) in patient_nums:
                        new_view_dataset.append(file)
                else:
                    if int(file["filename"].split("/")[-1][1:5]) not in patient_nums:
                        new_view_dataset.append(file)
            self.dataset[view]["dataset"] = new_view_dataset

        self._get_cumulatives()
        self._get_len()

    def print_statistics_per_patient(self):
        for i in range(0,len(self.dataset),2):
            for patient in range(len(self.dataset[i])):
                if "empty" in self.dataset[i][patient]:
                    print("File: empty")
                    continue
                f = h5py.File(self.dataset[i][patient], 'r')
                labels = np.array(f["labels"]).argmax(axis=1)
                c, counts = np.unique(labels,return_counts=True)
                s = "File: {} has {} windows with labels ".format(self.dataset[i][patient], len(labels))
                for i in range(len(c)):
                    s += "{}-{} ".format(c[i],f'{counts[i]:04}')
                print(s)

    def transform_images(self, images, num):
        aug_method = getattr(self.tf, self.aug[str(num)]["method"])
        # aug_method = globals()[self.aug[num]["method"]]
        for i in range(len(images)):
            # print("{}_{}".format(i,num))
            images[i] = aug_method(images[i], self.aug[str(num)], num)
        return images

class Sleep_Dataset_preloaded(Dataset):
    def __init__(self, config, file_dirs, num_views, data_augmentation= {}):
        self.prev_dataset = Sleep_Dataset_mat_huy(config, file_dirs, num_views)
        self.output, self.labels, self.init, self.ids, self.pre_loaded_idx = self.prev_dataset.preload_data()
        print("We have loaded up to sample {}".format(self.pre_loaded_idx))
        self.normalize = self.prev_dataset.normalize
        self.data_view_dir = self.prev_dataset.config.data_view_dir

        self.len = self.prev_dataset.__len__()

    def __getitem__(self, index):
        if index < self.pre_loaded_idx:
            return self.output[index], self.labels[index], self.init[index], self.ids[index]
        else:
            return self.prev_dataset.__getitem__(index=index)

    def __len__(self):
        return self.prev_dataset.__len__()

    def set_mean_std(self, mean, std):
        self.mean = mean
        self.std = std

        self.prev_dataset.set_mean_std(mean=mean, std=std)

        for i in range(len(self.output)):
            for mod in range(len(self.output[i])):
                if self.normalize:
                    if "time" in self.data_view_dir[mod][1][0]:
                        self.output[i][mod] = einops.rearrange(self.output[i][mod], "inner channels freq time -> inner time channels freq")
                        self.output[i][mod] = (self.output[i][mod] - self.mean[mod]) / self.std[mod]
                        self.output[i][mod] = einops.rearrange(self.output[i][mod], "inner time channels freq -> inner channels freq time")
                    if "stft" in self.data_view_dir[mod][1][0]:
                        self.output[i][mod] = einops.rearrange(self.output[i][mod], "inner channels freq time -> inner time channels freq")
                        self.output[i][mod] = (self.output[i][mod] - self.mean[mod]) / self.std[mod]
                        self.output[i][mod] = einops.rearrange(self.output[i][mod], "inner time channels freq -> inner channels freq time")

class SleepDataLoader_mat_huy():

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        sleep_dataset_train, sleep_dataset_val, sleep_dataset_test, sleep_dataset_total = self._get_datasets()

        self.get_norm_metrics(sleep_dataset_train=sleep_dataset_train, sleep_dataset_val=sleep_dataset_val, sleep_dataset_total=sleep_dataset_total, sleep_dataset_test=sleep_dataset_test)

        # shuffle_training_data =  False if self.config.seq_legth[0]>1 elif hasattr(self.config,"shuffle_train") self.config.shuffle_train else True

        # if self.config.dataset.seq_legth[0]>1: shuffle_training_data=False
        # elif  hasattr(self.config,"shuffle_train"): shuffle_training_data=self.config.dataset.shuffle_train
        # else: shuffle_training_data=True

        self.train_loader = torch.utils.data.DataLoader(sleep_dataset_train, batch_size=self.config.training_params.batch_size, num_workers=self.config.training_params.data_loader_workers,
                                                        pin_memory=self.config.training_params.pin_memory,
                                                        worker_init_fn=_init_fn)
        self.valid_loader = torch.utils.data.DataLoader(sleep_dataset_val, batch_size=self.config.training_params.test_batch_size,
                                                        shuffle=False, num_workers=self.config.training_params.data_loader_workers,
                                                        pin_memory=self.config.training_params.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(sleep_dataset_test, batch_size=self.config.training_params.test_batch_size,
                                                       shuffle=False, num_workers=self.config.training_params.data_loader_workers,
                                                       pin_memory=self.config.training_params.pin_memory)

        self.total_loader = torch.utils.data.DataLoader(sleep_dataset_total, batch_size=self.config.training_params.test_batch_size,
                                                       shuffle=False, num_workers=self.config.training_params.data_loader_workers,
                                                       pin_memory=self.config.training_params.pin_memory)

        if (self.config.statistics["print"] or  self.config.statistics["ce_weights"]["use"]) and not self.config.model.load_ongoing:
            self._statistics_mat()
        elif self.config.model.load_ongoing:
            print("We are loading weights")
            self.weights = np.zeros(self.config.num_classes)
        else:
            # self.weights = 1/self.train_loader.dataset.metrics["count_labels"]
            # norm = np.linalg.norm(self.weights)
            # self.weights = self.weights / norm

            #equal w
            self.weights = np.ones(self.config.num_classes)
            # self.weights[2] = 14.8*10
            # self.weights[1] = 7.67*10
            # self.weights[0] = 1.24
            # self.weights = np.log(self.weights)
        print("Weights are {}".format(self.weights))

    def load_metrics(self):

        print("Loading metrics from {}".format(self.config.normalization.dir))
        metrics_file = open(self.config.normalization.dir, "rb")
        self.metrics = pickle.load(metrics_file)

        if "train" in self.metrics.keys():
            self.metrics = self.metrics["train"]
        mean, std = {}, {}
        if self.config.normalization.metrics_type == "per_recording":
            mean = self.metrics["mean"]
            std = self.metrics["std"]
        if self.config.normalization.metrics_type == "sep_total_train_test":
            mean = self.metrics["mean"]
            std = self.metrics["std"]

        elif self.config.normalization.metrics_type == "total_dataset":
            for i, f in enumerate(self.config.data_view_dir):
                mod = f["data_type"] + "_" + f["mod"]
                mean[mod] = self.metrics["mean"][f["data_type"] + "_" + f["mod"]]
                std[mod] = self.metrics["std"][f["data_type"] + "_" + f["mod"]]
        elif self.config.normalization.metrics_type == "train_dataset":
                for i, f in enumerate(self.config.dataset.data_view_dir):
                    mod = f["data_type"] + "_" + f["mod"]
                    if "time" in mod:
                        mean[mod] = 0
                        std[mod] = 1
                    else:
                        mean[mod] = self.metrics["mean"][f["data_type"] + "_" + f["mod"]]
                        std[mod] = self.metrics["std"][f["data_type"] + "_" + f["mod"]]
        else:
            raise ValueError("Unknown metric type check config.normalization.metric_type")

                # mean[mod] = self.metrics["mean"]["{}_{}".format(f[1][0],f[1][1])]
                # std[mod] = self.metrics["std"]["{}_{}".format(f[1][0],f[1][1])]

        # These lines are only meant to be for sleepTransformers
        # mean, std = {}, {}
        # print(self.metrics.keys())
        # for i, f in enumerate(self.config.data_view_dir):
        #     mod = f["data_type"] + "_" + f["mod"]
        #     mean[mod] = self.metrics["train"]["mean"][f["data_type"] +"_"+ f["mod"]]
        #     std[mod] = self.metrics["train"]["std"][f["data_type"] +"_"+ f["mod"]]

            # mean[mod] = self.metrics["mean"]["{}_{}".format(f[1][0],f[1][1])]
            # std[mod] = self.metrics["std"]["{}_{}".format(f[1][0],f[1][1])]
        # print(self.metrics)
        return self.metrics

    def get_norm_metrics(self, sleep_dataset_train=None, sleep_dataset_val=None, sleep_dataset_test=None, sleep_dataset_total=None):

        if self.config.normalization.use and not self.config.model.load_ongoing:

            #calculate_metrics
            if not self.config.normalization.calculate_metrics:
                metrics = self.load_metrics()
                sleep_dataset_train.set_mean_std(metrics["mean"], metrics["std"])
                sleep_dataset_val.set_mean_std(metrics["mean"], metrics["std"])
                sleep_dataset_total.set_mean_std(metrics["mean"], metrics["std"])
                if self.config.normalization.metrics_type == "sep_total_train_test":
                    if "mean_test" not in metrics or "std_test" not in metrics:
                        raise Warning("We dont have metrics for the test set!")
                        sleep_dataset_test.set_mean_std(metrics["mean"], metrics["std"])
                    else:
                        sleep_dataset_test.set_mean_std(metrics["mean_test"], metrics["std_test"])
                else:
                    sleep_dataset_test.set_mean_std(metrics["mean"], metrics["std"])
                self.metrics = metrics

            else:
                if self.config.normalization.metrics_type == "sep_total_train_test":
                    mean_train, std_train = self.calculate_mean_std_mymat_total(sleep_dataset_train.dataset)
                    mean_test, std_test = self.calculate_mean_std_mymat_total(sleep_dataset_test.dataset)
                    sleep_dataset_train.set_mean_std(mean_train, std_train)
                    sleep_dataset_val.set_mean_std(mean_train, std_train)
                    sleep_dataset_total.set_mean_std(mean_train, std_train)

                    sleep_dataset_test.set_mean_std(mean_test, std_test)

                    self._save_metrics( {"mean": mean_train, "std": std_train, "mean_test":mean_test, "std_test":std_test})
                    self.metrics = {"mean": mean_train, "std": std_train, "mean_test":mean_test, "std_test":std_test}


                else:

                    if self.config.normalization.metrics_type == "per_recording":
                        mean, std = self.calculate_mean_std_mymat_perrecording(sleep_dataset_total.dataset)

                    elif self.config.normalization.metrics_type == "train_dataset":
                        mean, std = self.calculate_mean_std_mymat_total(sleep_dataset_train.dataset)

                    elif self.config.normalization.metrics_type == "total_dataset":
                        mean, std = self.calculate_mean_std_mymat_total(sleep_dataset_total.dataset)

                    else:
                        raise ValueError("Normalization metric_type is not valid")
                    sleep_dataset_train.set_mean_std(mean, std)
                    sleep_dataset_val.set_mean_std(mean, std)
                    sleep_dataset_total.set_mean_std(mean, std)
                    sleep_dataset_test.set_mean_std(mean, std)
                    self.metrics = {"mean": mean, "std": std}


        else:
            self.metrics = {"mean": None, "std": None}

    def load_metrics_ongoing(self, metrics):
        mean = metrics["mean"]
        std = metrics["std"]
        self.metrics = metrics
        self.train_loader.dataset.set_mean_std(mean, std)
        self.valid_loader.dataset.set_mean_std(mean, std)
        # self.test_loader.dataset.set_mean_std(mean, std)
        self.total_loader.dataset.set_mean_std(mean, std)

        if self.config.normalization.metrics_type == "sep_total_train_test":
            if "mean_test" not in metrics or "std_test" not in metrics:
                mean_test, std_test = self.calculate_mean_std_mymat_total(self.test_loader.dataset.dataset)
                self.test_loader.dataset.set_mean_std(mean_test, std_test)
            else:
                self.test_loader.dataset.set_mean_std(metrics["mean_test"], metrics["std_test"])
        else:
            self.test_loader.dataset.set_mean_std(mean, std)

    def calculate_mean_std_huy(self, dataset):

        mean, mean_sq, std, sum = {}, {}, {}, {}
        for i in range(len(self.config.data_view_dir)):
            mean[i],mean_sq[i],std[i],sum[i] = {}, {}, {}, 0

        #ONLINE CALCULATION
        # for view_i in range(0, 2 * len(self.config.data_view_dir), 2):
        #     for file_idx in tqdm(range(len(dataset[0])),"Mean-STD calc for {} modality".format((view_i//2) + 1)):
        #         f = h5py.File(dataset[view_i][file_idx], 'r')
        #         data = np.array(f["X2"])
        #         if sum[view_i]==0:
        #             mean[view_i] = data.mean(axis=1).mean(axis=1)
        #             mean_sq[view_i] = np.square(data).mean(axis=1).mean(axis=1)
        #         else:
        #             mean[view_i] = (mean[view_i]*sum[view_i] + data.mean(axis=1).mean(axis=1) * data.shape[1]*data.shape[2])/ (data.shape[1]*data.shape[2] + sum[view_i] )
        #             mean_sq[view_i] = (mean_sq[view_i] * sum[view_i] + np.square(data).mean(axis=1).mean(axis=1) * data.shape[1]*data.shape[2]) / (data.shape[1]*data.shape[2] + sum[view_i])
        #         sum[view_i] += data.shape[1]*data.shape[2]

        #OFFLINE 2 PASS CALC


        for view_i in range(0, 2 * len(self.config.data_view_dir), 2):
            pbar = tqdm(range(len(dataset[0])), "Mean calc for {} modality".format(int(view_i / 2) + 1))
            for file_idx in pbar:
                f = h5py.File(dataset[view_i][file_idx], 'r')
                data = np.array(f["X2"])
                sum_file = data.sum(axis=1).sum(axis=1)
                length = data.shape[1]*data.shape[2]

                if sum[view_i]==0:
                    mean[view_i] = sum_file
                else:
                    mean[view_i] += sum_file
                sum[view_i] += length
                pbar.set_description("Mean for file {} is {} ".format(file_idx+1, (sum_file/length).mean()))
                pbar.refresh()
            mean[view_i] /= sum[view_i] if sum[view_i] != 0 else 0

            for file_idx in tqdm(range(len(dataset[0])),"STD calc for {} modality".format((view_i/2) + 1)):
                f = h5py.File(dataset[view_i][file_idx], 'r')
                data = np.array(f["X2"])
                if file_idx==0:
                    std[view_i] = np.square(einops.rearrange(data, "f t batch -> batch t f") - mean[view_i]).sum(axis=0).sum(axis=0)
                else:
                    std[view_i] += np.square(einops.rearrange(data, "f t batch -> batch t f") - mean[view_i]).sum(axis=0).sum(axis=0)
                    # mean[view_i] = (mean[view_i]*sum[view_i] + data.mean(axis=1).mean(axis=1) * data.shape[1]*data.shape[2])/ (data.shape[1]*data.shape[2] + sum[view_i] )
                    # mean_sq[view_i] = (mean_sq[view_i] * sum[view_i] + np.square(data).mean(axis=1).mean(axis=1) * data.shape[1]*data.shape[2]) / (data.shape[1]*data.shape[2] + sum[view_i])
                # sum[view_i] += data.shape[1]*data.shape[2]
            std[view_i] = np.sqrt( std[view_i]/ sum[view_i]) if sum[view_i] != 0 else 0
            # mean_sq[view_i] = mean_sq[view_i]/sum[view_i]) if sum[view_i]) != 0 else 0

        return mean, std

    def _gather_metrics_total(self, metrics_scramble):
        metrics = {}
        metrics["mean"],  metrics["mean_sq"],  metrics["std"],  metrics["sum"] = {}, {}, {}, {}
        for i in range(len(self.config.dataset.data_view_dir)):
            mod = self.config.dataset.data_view_dir[i]["data_type"] + "_" + self.config.dataset.data_view_dir[i]["mod"]
            metrics["mean"][mod],  metrics["mean_sq"][mod],  metrics["std"][mod],  metrics["sum"][mod] = None, None, None, None
        varX = {}

        for i in range(len(self.config.dataset.data_view_dir)):
        # for view in dataset:

            mod = self.config.dataset.data_view_dir[i]["data_type"] + "_" + self.config.dataset.data_view_dir[i]["mod"]
            metrics["mean"][mod], metrics["sum"][mod], metrics["mean_sq"][mod], metrics["std"][mod] = {},{},{},{}
            print(self.config.dataset.data_view_dir[i]["num_ch"])
            for key in range(self.config.dataset.data_view_dir[i]["num_ch"]):
                for metrics_p in metrics_scramble:
                    if metrics_p == [] or (metrics_p["mean"][mod] and len(metrics_p["mean"][mod].keys())<1):
                        continue

                    if "ch_{}".format(key) in metrics_p["sum"][mod].keys():
                        if "ch_{}".format(key) not in metrics["sum"][mod]:
                            metrics["mean"][mod]["ch_{}".format(key)] = metrics_p["mean"][mod]["ch_{}".format(key)]
                            metrics["mean_sq"][mod]["ch_{}".format(key)] = metrics_p["mean_sq"][mod]["ch_{}".format(key)]
                            metrics["sum"][mod]["ch_{}".format(key)] = metrics_p["sum"][mod]["ch_{}".format(key)]
                        else:
                            metrics["mean"][mod]["ch_{}".format(key)] = (metrics["mean"][mod]["ch_{}".format(key)] * metrics["sum"][mod]["ch_{}".format(key)] + metrics_p["mean"][mod]["ch_{}".format(key)] * metrics_p["sum"][mod]["ch_{}".format(key)]) / (
                                        metrics_p["sum"][mod]["ch_{}".format(key)] + metrics["sum"][mod]["ch_{}".format(key)])
                            metrics["mean_sq"][mod]["ch_{}".format(key)] = (metrics["mean_sq"][mod]["ch_{}".format(key)] * metrics["sum"][mod]["ch_{}".format(key)] + metrics_p["mean_sq"][mod]["ch_{}".format(key)] * metrics_p["sum"][mod]["ch_{}".format(key)]) / (
                                        metrics_p["sum"][mod]["ch_{}".format(key)] + metrics["sum"][mod]["ch_{}".format(key)])
                            metrics["sum"][mod]["ch_{}".format(key)] += metrics_p["sum"][mod]["ch_{}".format(key)]
                varX["ch_{}".format(key)] = -np.multiply(metrics["mean"][mod]["ch_{}".format(key)], metrics["mean"][mod]["ch_{}".format(key)]) + metrics["mean_sq"][mod]["ch_{}".format(key)]
                metrics["std"][mod]["ch_{}".format(key)] = np.sqrt((varX["ch_{}".format(key)] * metrics["sum"][mod]["ch_{}".format(key)]) / (metrics["sum"][mod]["ch_{}".format(key)] - 1))
            # elif self.config.data_view_dir[i][1][0] == "time":
            #     for metrics_p in metrics_scramble:
            #         if metrics_p == []:
            #             continue
            #         if metrics["sum"][mod] == 0:
            #             metrics["mean"][mod] = metrics_p["mean"][mod]
            #             metrics["mean_sq"][mod] = metrics_p["mean_sq"][mod]
            #         else:
            #             metrics["mean"][mod] = (metrics["mean"][mod] * metrics["sum"][mod] + metrics_p["mean"][mod] * metrics_p["sum"][mod]) / (
            #                         metrics_p["sum"][mod] + metrics["sum"][mod])
            #             metrics["mean_sq"][mod] = (metrics["mean_sq"][mod] * metrics["sum"][mod] + metrics_p["mean_sq"][mod] * metrics_p["sum"][mod]) / (
            #                         metrics_p["sum"][mod] + metrics["sum"][mod])
            #         metrics["sum"][mod] += metrics_p["sum"][mod]
            #     varX = -np.multiply(metrics["mean"][mod], metrics["mean"][mod]) + metrics["mean_sq"][mod]
            #     metrics["std"][mod] = np.sqrt((varX * metrics["sum"][mod]) / (metrics["sum"][mod] - 1))
            #     raise NotImplementedError()
        return metrics
    def _gather_metrics_perrecording(self, metrics_scramble):

        metrics = {"mean":{}, "mean_sq":{}, "std": {}, "sum": {}}
        varX ={}
        for file_idx in range(len(self.config.dataset.data_view_dir[0])):
            varX[file_idx] = {}
            for i in range(len(self.config.dataset.data_view_dir)):
                mod = self.config.dataset.data_view_dir[i]["data_type"] + "_" + self.config.dataset.data_view_dir[i]["mod"]
                metrics["mean"][file_idx],  metrics["mean_sq"][file_idx],  metrics["std"][file_idx],  metrics["sum"][file_idx] = {}, {}, {}, {}
                metrics["mean"][file_idx][mod],  metrics["mean_sq"][file_idx][mod],  metrics["std"][file_idx][mod],  metrics["sum"][file_idx][mod] = {}, {}, {}, {}

        metrics = {"mean": {}, "mean_sq": {}, "std": {}, "sum": {}}
        for metrics_p in metrics_scramble:
            metrics["mean"].update(metrics_p["mean"])
            metrics["mean_sq"].update(metrics_p["mean"])
            metrics["std"].update(metrics_p["mean"])
            metrics["sum"].update(metrics_p["mean"])

        return metrics
    def _parallel_file_calculate_mean_std_mymat_total(self,dataset, file_idx):

        metrics = {}
        metrics["mean"],  metrics["mean_sq"],  metrics["std"],  metrics["sum"] = {}, {}, {}, {}
        for i in range(len(self.config.dataset.data_view_dir)):
            mod = self.config.dataset.data_view_dir[i]["data_type"] + "_" + self.config.dataset.data_view_dir[i]["mod"]
            metrics["mean"][mod],  metrics["mean_sq"][mod],  metrics["std"][mod],  metrics["sum"][mod] = {}, {}, {}, {}
            #
            # # ONLINE CALCULATION
            # pbar = tqdm(range(len(dataset[0])), "Mean calc for {} modality".format(int(view_i / 2) + 1))
            # for file_idx in pbar:
        for view in dataset:
            filename = dataset[view]["dataset"][file_idx]["filename"]
            if "empty" in filename:
                continue
            f = h5py.File(filename, 'r')
            # f = zarr.open(dataset[view_i][file_idx], 'r')

            if  dataset[view]["data_type"] == "time":
                data = np.array(f["X1"])
                if self.config.dataset.huy_data:
                    metrics["mean"][view]["ch_0"]  = data.mean()
                    metrics["mean_sq"][view]["ch_0"]  = np.square(data).mean()
                    metrics["sum"][view]["ch_0"] = data.shape[0] * data.shape[1]
                else:
                    raise NotImplementedError()
            elif  dataset[view]["data_type"] == "stft":
                if self.config.dataset.huy_data:
                    data = f["X2"]
                    data = np.array(data)
                    if len(data.shape)<3: continue
                    metrics["mean"][view]["ch_0"]  = data.mean(axis=(1, 2))
                    metrics["mean_sq"][view]["ch_0"]  = np.square(data).mean(axis=(1, 2))
                    metrics["sum"][view]["ch_0"] = data.shape[1] * data.shape[2]
                else:
                    data = {}
                    for key in range(dataset[view]["num_ch"]):
                        if "X2_ch_{}".format(key) in f.keys():
                            this_channel = np.expand_dims(f["X2_ch_{}".format(key)], axis=1)
                            if len(this_channel.shape)!=4:
                                print("Channel X2_ch_{} is missing".format(key))
                                continue
                            data["ch_{}".format(key)] = this_channel
                        else:
                            print("Channel X2_ch{} is missing".format(key))

                    # file_mean, file_mean_sq, file_length = {}, {}, {}
                    for i in data.keys():
                        if len(data[i].shape) <3: continue
                        metrics["mean"][view][i] = data[i].mean(axis=(0, 3))
                        metrics["mean_sq"][view][i] = np.square(data[i]).mean(axis=(0, 3))
                        metrics["sum"][view][i] = data[i].shape[0] * data[i].shape[3]

                    # data = f["X2_ch_0"]
                    # data = np.expand_dims(np.array(data),axis=1)
                    # data = np.concatenate([np.expand_dims(data[key],axis=1) for key in data.keys()], axis=1)
                    #Dim 0 should be th
                    # file_mean = data.mean(axis=(0, 3))
                    # file_mean_sq = np.square(data).mean(axis=(0, 3))
                    # file_length = data.shape[0] * data.shape[3]

            # metrics["mean"][mod] = file_mean
            # metrics["mean_sq"][mod] = file_mean_sq
            # metrics["sum"][mod] = file_length

        return metrics
    def _parallel_file_calculate_mean_std_mymat_perrecording(self,dataset, file_idx):

        metrics = {"mean":{file_idx:{}}, "mean_sq":{file_idx: {}}, "std": {file_idx: {}}, "sum": {file_idx: {}}}

        for i in range(len(self.config.data_view_dir)):
            mod = self.config.data_view_dir[i]["data_type"] + "_" + self.config.data_view_dir[i]["mod"]
            metrics["mean"][file_idx][mod],  metrics["mean_sq"][file_idx][mod],  metrics["std"][file_idx][mod],  metrics["sum"][file_idx][mod] = {}, {}, {}, {}
            #
            # # ONLINE CALCULATION
            # pbar = tqdm(range(len(dataset[0])), "Mean calc for {} modality".format(int(view_i / 2) + 1))
            # for file_idx in pbar:
        for view_i in range(0, 2 * len(self.config.data_view_dir), 2):
            if "empty" in dataset[view_i][file_idx]:
                continue
            f = h5py.File(dataset[view_i][file_idx], 'r')
            # f = zarr.open(dataset[view_i][file_idx], 'r')
            mod = self.config.data_view_dir[int(view_i/2)]["data_type"] + "_" + self.config.data_view_dir[int(view_i/2)]["mod"]
            if  self.config.data_view_dir[int(view_i/2)]["data_type"] == "time":
                data = np.array(f["X1"])
                if self.config.huy_data:
                    file_mean = data.mean()
                    file_mean_sq = np.square(data).mean()
                    file_length = data.shape[0] * data.shape[1]
                else:
                    raise NotImplementedError()
            elif  self.config.data_view_dir[int(view_i/2)]["data_type"] == "stft":
                if self.config.huy_data:
                    data = f["X2"]
                    data = {"ch_{}".format(0): np.array(data)}

                    for i in data.keys():
                        if len(data[i].shape) <3: continue
                        metrics["mean"][file_idx][mod][i] = data[i].mean(axis=(1, 2))
                        metrics["mean_sq"][file_idx][mod][i] = np.square(data[i]).mean(axis=(1, 2))
                        metrics["sum"][file_idx][mod][i] = data[i].shape[1] * data[i].shape[2]
                else:
                    data = {}
                    for key in range(self.config.data_view_dir[int(view_i/2)]["num_ch"]):
                        if "X2_ch_{}".format(key) in f.keys():
                            this_channel = np.expand_dims(f["X2_ch_{}".format(key)], axis=1)
                            if len(this_channel.shape)!=4:
                                print("Channel X2_ch_{} is missing".format(key))
                                continue
                            data["ch_{}".format(key)] = this_channel
                        else:
                            print("Channel X2_ch{} is missing".format(key))

                    # file_mean, file_mean_sq, file_length = {}, {}, {}
                    for i in data.keys():
                        if len(data[i].shape) <3: continue
                        metrics["mean"][file_idx][mod][i] = data[i].mean(axis=(0, 3))
                        metrics["mean_sq"][file_idx][mod][i] = np.square(data[i]).mean(axis=(0, 3))
                        metrics["sum"][file_idx][mod][i] = data[i].shape[0] * data[i].shape[3]

        return metrics
    def _save_metrics(self, metrics):
        try:
            metrics_file = open(self.config.normalization.dir, "wb")
            pickle.dump(metrics, metrics_file)
            metrics_file.close()
            print("Metrics saved!")
        except:
            print("Error on saving metrics")

    def calculate_mean_std_mymat_total(self, dataset):

        num_cores = self.config.training_params.data_loader_workers
        metrics_scramble = Parallel(n_jobs=num_cores)(delayed(self._parallel_file_calculate_mean_std_mymat_total)(dataset, file_idx) for file_idx in tqdm(range(len(dataset[list(dataset.keys())[0]]["dataset"])), "Mean std calculations")) #len(dataset[0])
        # metrics_scramble  = []
        # for file_idx in tqdm(range(78), "Mean std calculations"):
        #     metrics_scramble.append(self._parallel_file_calculate_mean_std_mymat_total(dataset, file_idx))

        metrics = self._gather_metrics_total(metrics_scramble)
        if self.config.normalization.save_metrics:
            self._save_metrics(metrics)
        return metrics["mean"], metrics["std"]

    def calculate_mean_std_mymat_perrecording(self, dataset):

        num_cores = 20
        metrics_scramble = Parallel(n_jobs=num_cores)(delayed(self._parallel_file_calculate_mean_std_mymat_perrecording)(dataset, file_idx) for file_idx in tqdm(range(len(dataset[0])), "Mean std calculations")) #len(dataset[0])
        # metrics_scramble  = []
        # for file_idx in tqdm(range(10), "Mean std calculations"):
        #     metrics_scramble.append(self._parallel_file_calculate_mean_std_mymat_perrecording(dataset, file_idx))

        metrics = self._gather_metrics_perrecording(metrics_scramble)
        if self.config.normalization.save_metrics:
            self._save_metrics(metrics)
        return metrics["mean"], metrics["std"]

        # #ONLINE CALCULATION
        # for view_i in range(0, 2 * len(self.config.data_view_dir), 2):
        #     pbar = tqdm(range(len(dataset[0])), "Mean calc for {} modality".format(int(view_i / 2) + 1))
        #     for file_idx in pbar:
        #         f = h5py.File(dataset[view_i][file_idx], 'r')
        #         data = np.array(f["X2"])[:,0,:,:]
        #         file_mean = data.mean(axis=(0, 2))
        #         file_mean_sq = np.square(data).mean(axis=(0, 2))
        #         file_length = data.shape[0]*data.shape[2]
        #         if sum[view_i]==0:
        #             mean[view_i] = file_mean
        #             mean_sq[view_i] = file_mean_sq
        #         else:
        #             mean[view_i] = (mean[view_i]*sum[view_i] + file_mean*file_length)/ (file_length + sum[view_i])
        #             mean_sq[view_i] = (mean_sq[view_i] * sum[view_i] + file_mean_sq * file_length) / (file_length + sum[view_i])
        #         sum[view_i] += file_length
        #         pbar.set_description("Mean for file {} is {} ".format(file_idx+1, mean[view_i].mean()))
        #         pbar.refresh()
        #     varX = -np.multiply(mean[view_i], mean[view_i]) + mean_sq[view_i]
        #     std[view_i] = np.sqrt((varX*sum[view_i])/(sum[view_i]-1))

        #OFFLINE 2 PASS CALC

        # mean, mean_sq, std, sum = {}, {}, {}, {}
        # for i in range(len(self.config.data_view_dir)):
        #     mean[i],mean_sq[i],std[i],sum[i] = None, None, None, 0
        #
        #
        # for view_i in range(0, 2 * len(self.config.data_view_dir), 2):
        #     pbar = tqdm(range(len(dataset[0])), "Mean calc for {} modality".format(int(view_i / 2) + 1))
        #     for file_idx in pbar:
        #         f = h5py.File(dataset[view_i][file_idx], 'r')
        #         data = np.array(f["X2"])
        #         sum_file = data[:,0,:,:].sum(axis=0).sum(axis=1)
        #         length = data.shape[0]*data.shape[3]
        #
        #         if sum[view_i]==0:
        #             mean[view_i] = sum_file
        #         else:
        #             mean[view_i] += sum_file
        #         sum[view_i] += length
        #         pbar.set_description("Mean for file {} is {} ".format(file_idx+1, (sum_file/length).mean()))
        #         pbar.refresh()
        #     mean[view_i] /= sum[view_i] if sum[view_i] != 0 else 0
        #     for file_idx in tqdm(range(len(dataset[0])),"STD calc for {} modality".format((view_i/2) + 1)):
        #         f = h5py.File(dataset[view_i][file_idx], 'r')
        #         data = np.array(f["X2"])
        #         if file_idx==0:
        #             std[view_i] = np.square(einops.rearrange(data[:,0,:,:], "batch f t-> batch t f") - mean[view_i]).sum(axis=0).sum(axis=0)
        #         else:
        #             std[view_i] += np.square(einops.rearrange(data[:,0,:,:], "batch f t-> batch t f") - mean[view_i]).sum(axis=0).sum(axis=0)
        #             # mean[view_i] = (mean[view_i]*sum[view_i] + data.mean(axis=1).mean(axis=1) * data.shape[1]*data.shape[2])/ (data.shape[1]*data.shape[2] + sum[view_i] )
        #             # mean_sq[view_i] = (mean_sq[view_i] * sum[view_i] + np.square(data).mean(axis=1).mean(axis=1) * data.shape[1]*data.shape[2]) / (data.shape[1]*data.shape[2] + sum[view_i])
        #         # sum[view_i] += data.shape[1]*data.shape[2]
        #     std[view_i] = np.sqrt( std[view_i]/ sum[view_i]) if sum[view_i] != 0 else 0
        #     print(mean[view_i].mean())
        #     print(std[view_i].mean())
        #     # mean_sq[view_i] = mean_sq[view_i]/sum[view_i]) if sum[view_i]) != 0 else 0

    # def _save_mean_std_mymat(self, dataset):
    #     ch_per_mod = [2,2,1]
    #     metrics = {}
    #     metrics_list = ["mean_stft", "mean_sq_stft", "std_stft", "sum_stft"]
    #     for set in ["train"]:
    #         metrics[set] = {}
    #         for m in metrics_list:
    #             metrics[set][m] = {}
    #             for mod in range(len(self.config.data_view_dir)):
    #                 metrics[set][m][mod] = {}
    #                 for ch in range(ch_per_mod[mod]):
    #                     metrics[set][m][mod][ch] = np.zeros(129) if "stft" in m and "sum" not in m else 0
    #
    #     # mean, mean_sq, std, sum = {}, {}, {}, {}
    #     # for i in range(len(self.config.data_view_dir)):
    #     #     mean[i],mean_sq[i],std[i],sum[i] = None, None, None, 0
    #
    #     #ONLINE CALCULATION
    #     for view_i in range(0, 2 * len(self.config.data_view_dir), 2):
    #         mod = int(view_i / 2)
    #         pbar = tqdm(range(len(dataset[0])), "Mean calc for {} modality".format(mod+1))
    #         for file_idx in pbar:
    #             for ch in range(ch_per_mod[mod]):
    #                 f = h5py.File(dataset[view_i][file_idx], 'r')
    #                 data = np.array(f["X2"])
    #                 data_ch = data[:, ch, :, :]
    #                 file_mean = data_ch.mean(axis=(0, 2))
    #                 file_mean_sq = np.square(data_ch).mean(axis=(0, 2))
    #                 file_length = data_ch.shape[0]*data_ch.shape[2]
    #                 if metrics["train"]["sum_stft"][mod][ch]==0:
    #                     metrics["train"]["mean_stft"][mod][ch] = file_mean
    #                     metrics["train"]["mean_sq_stft"][mod][ch] = file_mean_sq
    #                 else:
    #                     metrics["train"]["mean_stft"][mod][ch] = (metrics["train"]["mean_stft"][mod][ch]*metrics["train"]["sum_stft"][mod][ch] + file_mean*file_length)/ (file_length + metrics["train"]["sum_stft"][mod][ch])
    #                     metrics["train"]["mean_sq_stft"][mod][ch] = (metrics["train"]["mean_sq_stft"][mod][ch] * metrics["train"]["sum_stft"][mod][ch] + file_mean_sq * file_length) / (file_length + metrics["train"]["sum_stft"][mod][ch])
    #                 metrics["train"]["sum_stft"][mod][ch] += file_length
    #                 pbar.set_description("Mean for file {} is {} ".format(file_idx+1, metrics["train"]["mean_stft"][mod][ch].mean()))
    #                 pbar.refresh()
    #         for ch in range(ch_per_mod[mod]):
    #             varX = -np.multiply(metrics["train"]["mean_stft"][mod][ch], metrics["train"]["mean_stft"][mod][ch]) + metrics["train"]["mean_sq_stft"][mod][ch]
    #             metrics["train"]["std_stft"][mod][ch] = np.sqrt((varX*metrics["train"]["sum_stft"][mod][ch])/(metrics["train"]["sum_stft"][mod][ch]-1))
    #
    #     metrics_file = open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/V2_shhs1_mat/metrics_v2_mat.pkl", "wb")
    #     pickle.dump(metrics, metrics_file)
    #     metrics_file.close()
    #     print("Metrics saved!")
    #     #OFFLINE 2 PASS CALC
    #
    #     # mean, mean_sq, std, sum = {}, {}, {}, {}
    #     # for i in range(len(self.config.data_view_dir)):
    #     #     mean[i],mean_sq[i],std[i],sum[i] = None, None, None, 0
    #     #
    #     #
    #     # for view_i in range(0, 2 * len(self.config.data_view_dir), 2):
    #     #     pbar = tqdm(range(len(dataset[0])), "Mean calc for {} modality".format(int(view_i / 2) + 1))
    #     #     for file_idx in pbar:
    #     #         f = h5py.File(dataset[view_i][file_idx], 'r')
    #     #         data = np.array(f["X2"])
    #     #         sum_file = data[:,0,:,:].sum(axis=0).sum(axis=1)
    #     #         length = data.shape[0]*data.shape[3]
    #     #
    #     #         if sum[view_i]==0:
    #     #             mean[view_i] = sum_file
    #     #         else:
    #     #             mean[view_i] += sum_file
    #     #         sum[view_i] += length
    #     #         pbar.set_description("Mean for file {} is {} ".format(file_idx+1, (sum_file/length).mean()))
    #     #         pbar.refresh()
    #     #     mean[view_i] /= sum[view_i] if sum[view_i] != 0 else 0
    #     #     for file_idx in tqdm(range(len(dataset[0])),"STD calc for {} modality".format((view_i/2) + 1)):
    #     #         f = h5py.File(dataset[view_i][file_idx], 'r')
    #     #         data = np.array(f["X2"])
    #     #         if file_idx==0:
    #     #             std[view_i] = np.square(einops.rearrange(data[:,0,:,:], "batch f t-> batch t f") - mean[view_i]).sum(axis=0).sum(axis=0)
    #     #         else:
    #     #             std[view_i] += np.square(einops.rearrange(data[:,0,:,:], "batch f t-> batch t f") - mean[view_i]).sum(axis=0).sum(axis=0)
    #     #             # mean[view_i] = (mean[view_i]*sum[view_i] + data.mean(axis=1).mean(axis=1) * data.shape[1]*data.shape[2])/ (data.shape[1]*data.shape[2] + sum[view_i] )
    #     #             # mean_sq[view_i] = (mean_sq[view_i] * sum[view_i] + np.square(data).mean(axis=1).mean(axis=1) * data.shape[1]*data.shape[2]) / (data.shape[1]*data.shape[2] + sum[view_i])
    #     #         # sum[view_i] += data.shape[1]*data.shape[2]
    #     #     std[view_i] = np.sqrt( std[view_i]/ sum[view_i]) if sum[view_i] != 0 else 0
    #     #     print(mean[view_i].mean())
    #     #     print(std[view_i].mean())
    #     #     # mean_sq[view_i] = mean_sq[view_i]/sum[view_i]) if sum[view_i]) != 0 else 0
    #
    #     return mean, std

    def _get_datasets(self):

        views = {}
        for i in self.config.dataset.data_view_dir:
            i["list_dir"] = self.config.dataset.data_roots + "/" + i["list_dir"]
            views[i["data_type"] + "_" + i["mod"]] = i
        views = self._read_dirs_mat(views)
        train_views, val_views, test_views = self._split_data_mat(views)

        train_dataset = Sleep_Dataset_mat_huy(config=self.config, views=train_views, set_name="train") if not self.config.dataset.cache_datasets["train"] else Sleep_Dataset_preloaded(self.config, train_views)
        valid_dataset = Sleep_Dataset_mat_huy(config=self.config, views=val_views, set_name="val") if not self.config.dataset.cache_datasets["valid"] else Sleep_Dataset_preloaded(self.config, val_views)
        test_dataset = Sleep_Dataset_mat_huy(config=self.config, views=test_views, set_name="test") if not self.config.dataset.cache_datasets["test"] else Sleep_Dataset_preloaded(self.config, test_views)
        total_dataset = Sleep_Dataset_mat_huy(config=self.config, views=views, set_name="total")

        return train_dataset, valid_dataset, test_dataset, total_dataset

    def _read_dirs_mat(self, view_dirs):

        for view in view_dirs:
            list_dir = view_dirs[view]["list_dir"]
            dataset = []
            with open(list_dir) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\n')
                for j, row in enumerate(csv_reader):
                    dt = row[0].split("-")
                    dataset.append({"filename": dt[0], "len_windows": dt[1]})
            view_dirs[view]["dataset"] = dataset
        return view_dirs

    def _unique_dataloader(self,dataset):
        label_description = "label" if self.config.dataset.huy_data else "labels"
        counts = {}
        for view in dataset:
            counts[view]= np.zeros(self.config.model.args.num_classes)
        #Calculate weights for unbalanced classes
        total_classes = np.arange(self.config.model.args.num_classes)
        for view in dataset:
            for file in tqdm(dataset[view]["dataset"],"Counting"):
                # f = h5py.File(file,"r")
                if "empty" in file["filename"]:
                    continue
                f = h5py.File(file["filename"],"r")["label"] if self.config.dataset.huy_data else h5py.File(file["filename"],"r")["labels"]

                labels = np.array(f).squeeze()
                if len(labels.shape) > 1 and labels.shape[1]==2:
                    labels = labels[:,0].squeeze()
                if self.config.dataset.huy_data:
                    labels = labels - 1
                if len(labels.shape)>1:
                    labels = labels.argmax(axis=1)
                classes, c = np.unique(labels,return_counts=True)

                # #This is in case we only keep the patients that have all labels
                # if len(c)<self.config.num_classes:
                #     # print(file)
                #     continue
                for i, cl in enumerate(classes):
                    counts[view][int(cl)] += c[int(i)]


        print("We are keeping all patients, change this for SHHS")

        return total_classes, counts

    def _statistics_mat(self):

        total = []
        classes, counts = self._unique_dataloader(self.total_loader.dataset.dataset)
        total.append(["Total", classes, counts])
        classes, counts = self._unique_dataloader(self.train_loader.dataset.dataset)
        total.append(["Training", classes, counts])
        v_classes, v_counts = self._unique_dataloader(self.valid_loader.dataset.dataset)
        total.append(["Validation", v_classes, v_counts])
        if self.config.training_params.use_test_set:
            t_classes, t_counts = self._unique_dataloader(self.test_loader.dataset.dataset)
            total.append(["Test", t_classes, t_counts])
        if self.config.statistics["print"]:
            for label, cl, c in total:
                print("In {} set we got".format(label))
                for view in c:
                    s = "{}: ".format(view)
                    for i in range(len(c[view])):
                        s = s + "Label {} : {} ".format(cl[i], int(c[view][i]))
                    print(s)
        if self.config.statistics["ce_weights"]["use"]:
            temperature = self.config.statistics["ce_weights"]["temp"]*counts[list(counts.keys())[0]].sum()
            self.weights = counts[list(counts.keys())[0]].sum()/(counts[list(counts.keys())[0]]+temperature)
            norm = np.linalg.norm(self.weights)
            self.weights = self.weights / norm
        else:
            self.weights = np.ones(self.config.model.args.num_classes)
        print(self.weights)

    def _split_data_mat(self, views):
        dirs_test = np.array([])
        if (self.config.dataset.data_split.split_method == "patients_test"):
            dirs_train, dirs_val, dirs_test = self._split_patients_test(views, self.config.dataset.data_split.val_split_rate, self.config.dataset.data_split.test_split_rate )
        elif (self.config.dataset.data_split.split_method == "patients_folds"):
            dirs_train, dirs_val, dirs_test = self._split_patients_folds(views, self.config.dataset.data_split.folds_num, self.config.dataset.data_split.fold, self.config.dataset.data_split.val_split_rate)
        elif (self.config.dataset.data_split.split_method == "patients_huy"):
            print("We are splitting dataset by huy splits")
            dirs_train, dirs_val, dirs_test = self._split_patients_huy(views, 0)
        elif (self.config.dataset.data_split.split_method == "patients_sleepyco"):
            print("We are splitting dataset by sleepyco splits")
            dirs_train, dirs_val, dirs_test = self._split_patients_sleepyco(views, self.config.dataset.data_split.folds_num, self.config.dataset.data_split.fold)
        elif (self.config.dataset.data_split.split_method == "neonatal"):
            dirs_train, dirs_val, dirs_test = self._split_patients_neonatal(views,
                                                                            split_rate_val=self.config.dataset.data_split.val_split_rate,
                                                                            neonatal_test_views=self.config.dataset.data_split.neonatal_test_views)
        else:
            raise ValueError("No splitting method named {} exists.".format(self.config.dataset.data_split.split_method))
        return dirs_train, dirs_val, dirs_test

    def _split_patients_neonatal(self, dirs_train_whole, split_rate_val, neonatal_test_views):

        views_test = [self.config.data_roots + "/" + i[0] for i in neonatal_test_views]
        dirs_test, _ = self._read_dirs_mat(views_test)


        patient_names = np.unique(np.array([i.split("/")[-1][:5] for i in dirs_train_whole[0]]))
        train_val = train_test_split(patient_names, test_size=split_rate_val, random_state=self.config.seed)

        train_idx, val_idx = [],[]
        for index, file_name in enumerate(dirs_train_whole[0]):
            if file_name.split("/")[-1][:5] in train_val[0]:
                train_idx.append(index)
            else:
                val_idx.append(index)
        print("Train {} validation {} test {}".format(len(train_idx),len(val_idx),len(dirs_test[0])))
        split_dict = {"train":train_idx,"valid":val_idx}

        return np.array(dirs_train_whole[:,split_dict["train"]]), np.array(dirs_train_whole[:,split_dict["valid"]]), np.array(dirs_test)

    def _split_patients_test(self, dirs_train_whole, split_rate_val, split_rate_test):
        patient_names = np.array([i["filename"].split("/")[-1][:5] for i in dirs_train_whole[list(dirs_train_whole.keys())[0]]["dataset"]])
        trainval_test = train_test_split(patient_names, test_size=split_rate_test, random_state=self.config.training_params.seed)
        train_val = train_test_split(trainval_test[0], test_size=split_rate_val, random_state=self.config.training_params.seed)

        train_views = copy.deepcopy(dirs_train_whole)
        val_views = copy.deepcopy(dirs_train_whole)
        test_views = copy.deepcopy(dirs_train_whole)

        for view in dirs_train_whole:
            num_difference, prev = 0, -1
            train_dataset, val_dataset, test_dataset = [], [], []
            for file in dirs_train_whole[view]["dataset"]:
                patient_num = file["filename"].split("/")[-1][:5]
                if patient_num in trainval_test[1]:
                    test_dataset.append(file)
                elif patient_num in train_val[0]:
                    train_dataset.append(file)
                elif patient_num in train_val[1]:
                    val_dataset.append(file)
                else:
                    raise Warning("Splitting is not going well, some patient has no house. \n {}".format(file))

            if "tom_subset" in self.config.dataset.data_split and self.config.dataset.data_split.tom_subset:
                train_dataset = train_dataset[0:10]
            train_views[view]["dataset"] = train_dataset
            val_views[view]["dataset"] = val_dataset
            test_views[view]["dataset"] = test_dataset

            print("{} Train {} validation {} test {}".format(view, len(train_views[view]["dataset"]), len(val_views[view]["dataset"]), len(test_views[view]["dataset"])))

        # with open('/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/shhs1_mat.pickle', 'wb') as f:
        #     pickle.dump(split_dict, f)

        # with open('/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/seizit1_full_random_split_mat.pickle', 'rb') as f:
        #     split_dict = pickle.load(f)

        return train_views, val_views, test_views

        # return np.array(dirs_train_whole[:,split_dict["train"]]), np.array(dirs_train_whole[:,split_dict["valid"]]), np.array(dirs_train_whole[:,split_dict["test"]])

    def _split_patients_folds(self, dirs_train_whole, folds_num, fold, split_rate_val):

        patient_list = [ int(file["filename"].split("/")[-1][1:5]) for view in dirs_train_whole for file in dirs_train_whole[view]["dataset"] if file["filename"].split("/")[-1] != "empty"]
        patient_names = np.unique(patient_list)


        foldsplits = list(KFold(n_splits=folds_num, shuffle=True, random_state=self.config.training_params.seed).split(patient_names))[fold]
        trainval_val = train_test_split(foldsplits[0], test_size=split_rate_val, random_state=self.config.training_params.seed)
        train_idx, val_idx, test_idx = patient_names[trainval_val[0]], patient_names[trainval_val[1]], patient_names[foldsplits[1]]

        train_views = copy.deepcopy(dirs_train_whole)
        val_views = copy.deepcopy(dirs_train_whole)
        test_views = copy.deepcopy(dirs_train_whole)

        for view in dirs_train_whole:
            train_dataset, val_dataset, test_dataset = [], [], []
            for file in dirs_train_whole[view]["dataset"]:
                patient_num = int(file["filename"].split("/")[-1][1:5])
                if patient_num in train_idx:
                    train_dataset.append(file)
                elif patient_num in val_idx:
                    val_dataset.append(file)
                elif patient_num in test_idx:
                    test_dataset.append(file)
                else:
                    raise Warning("Splitting is not going well, some patient has no house. \n {}".format(file))

            train_views[view]["dataset"] = train_dataset
            val_views[view]["dataset"] = val_dataset
            test_views[view]["dataset"] = test_dataset

        print("Train {} validation {} test {} in fold {}/{}".format(len(train_idx), len(val_idx), len(test_idx), fold, folds_num-1))
        split_dict = {"train":train_views,"test":test_idx,"valid":val_views}
        # with open('/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/shhs1_mat.pickle', 'wb') as f:
        #     pickle.dump(split_dict, f)

        # with open('/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/seizit1_full_random_split_mat.pickle', 'rb') as f:
        #     split_dict = pickle.load(f)

        return train_views, val_views, test_views

    def _split_patients_huy(self, dirs_train_whole, fold):
        # patient_names = np.unique(np.array([i.split("/")[-1][:5] for i in dirs_train_whole[0]]))
        # # cross_val_run = pickle.load(open(self.config.folds_file, "rb"))
        f = loadmat(self.config.dataset.data_split.folds_file)
        f["train_sub"] = f["train_sub"].squeeze() - 1
        f["eval_sub"] = f["eval_sub"].squeeze() - 1
        f["test_sub"] = f["test_sub"].squeeze() - 1


        train_views = copy.deepcopy(dirs_train_whole)
        val_views = copy.deepcopy(dirs_train_whole)
        test_views = copy.deepcopy(dirs_train_whole)

        #WE need a running difference between the numbers of huy and ours. When we skip a patient we skip a number, huy doesnt.
        for view in dirs_train_whole:
            num_difference, prev = 0, -1
            train_dataset, val_dataset, test_dataset = [], [], []
            for file in dirs_train_whole[view]["dataset"]:
                patient_num = int(file["filename"].split("/")[-1][1:5])
                num_difference += patient_num - prev - 1
                prev = patient_num
                if patient_num - num_difference in f["train_sub"]:
                    train_dataset.append(file)
                elif patient_num - num_difference in f["eval_sub"]:
                    val_dataset.append(file)
                elif patient_num - num_difference in f["test_sub"]:
                    test_dataset.append(file)
                else:
                    raise Warning("Splitting is not going well, some patient has no house. \n {}".format(file))

            if "tom_subset" in self.config.dataset.data_split and self.config.dataset.data_split.tom_subset:
                train_dataset = train_dataset[0:10]
            train_views[view]["dataset"] = train_dataset
            val_views[view]["dataset"] = val_dataset
            test_views[view]["dataset"] = test_dataset

        # for index, file_name in dirs_train_whole:
        #     patient_num = int(file_name.split("/")[-1][1:5])
        #     num_difference += patient_num - prev - 1
        #     prev = patient_num
        #     if patient_num - num_difference in f["train_sub"]:
        #         train_idx.append(index)
        #     elif patient_num - num_difference in f["eval_sub"]:
        #         val_idx.append(index)
        #     elif patient_num - num_difference in f["test_sub"]:
        #         test_idx.append(index)
        #     else:
        #         raise Warning("Splitting is not going well, some patient has no house. \n {}".format(file_name))
        return train_views, val_views, test_views

    def _split_patients_sleepyco(self, dirs_train_whole, folds_num, fold):

        import pickle
        file = open(self.config.dataset.data_split.folds_file, "rb")
        patients_split = pickle.load(file)
        file.close()

        this_fold_split = patients_split[fold]

        train_views = copy.deepcopy(dirs_train_whole)
        val_views = copy.deepcopy(dirs_train_whole)
        test_views = copy.deepcopy(dirs_train_whole)

        for view in dirs_train_whole:
            train_dataset, val_dataset, test_dataset = [], [], []
            for file in dirs_train_whole[view]["dataset"]:
                patient_num = file["filename"].split("/")[-3]
                if patient_num in this_fold_split["train"]:
                    train_dataset.append(file)
                elif patient_num in this_fold_split["val"]:
                    val_dataset.append(file)
                elif patient_num in this_fold_split["test"]:
                    test_dataset.append(file)
                else:
                    raise Warning("Splitting is not going well, some patient has no house. \n {}".format(file))

            train_views[view]["dataset"] = train_dataset
            val_views[view]["dataset"] = val_dataset
            test_views[view]["dataset"] = test_dataset

        print("Train {} validation {} test {} in fold {}/{}".format(len(this_fold_split["train"]), len(this_fold_split["val"]), len(this_fold_split["test"]), fold, folds_num-1))

        return train_views, val_views, test_views

