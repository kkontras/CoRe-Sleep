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
# import zarr
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
import logging
import easydict
from os.path import join as pjoin
import logging

nested_dict = lambda: defaultdict(nested_dict)

class Sleep_Dataset(Dataset):

    def __init__(self, config: easydict.EasyDict, views: dict, set_name: str):
        """
        :param config: config file
        :param views: dictionary of views, keys are the view names and values are list of dictionaries containing the data filenames and length of epochs for each patient.
        :param set_name: name of the set, e.g. "train", "val", "test"
        """
        super()
        self.dataset = views
        self.views = list(self.dataset.keys())
        self.config = config
        self.set_name = set_name

        self._init_attributes()

        # Filter the windows to get a subset of the data
        if self.filter_patients["use_type"]:
            self._find_list_of_patients()
            self.broken_mod_dict = self._get_broken_modalities(filename=self.config.dataset.broken_patients_filepath)

        self._get_cumulatives()

    def _init_attributes(self):
        self.num_views = len(self.dataset)
        self.normalize = True
        self.outer_seq_length = self.config.dataset.outer_seq_length
        self.filter_patients = self.config.dataset.filter_patients[self.set_name]

    def _get_cumulatives(self):
        """
        Create the self.cumulatives dict which indicates at each idx which file we will open and which segments we will take.
        """
        view = list(self.dataset.keys())[0] # the first view, any should work
        self.cumulatives = {"lengths":[0], "files":{}}
        for file_idx in range(len(self.dataset[view]["dataset"])):
            file = self.dataset[view]["dataset"][file_idx]
            patient_num = int(file["filename"].split("/")[-1][1:5])
            file_len = int(file["len_windows"])

            if self.filter_patients["use_type"] == "include_only_skipped":
                self._single_patient_cumulative_includeskipped(patient_num, file_len, file_idx)
            elif self.filter_patients["use_type"] == "subsample":
                if not self.filter_patients["whole_patient"]:
                    raise Warning("Whole patients is not true which might lead to different sampling rather the desired one. Possibly disclude patients with both modalities available")
                self._single_patient_cumulative_includeskipped(patient_num, file_len, file_idx)
            else:
                self._single_patient_cumulative_full(patient_num, file_len, file_idx)

    def set_mean_std(self, mean, std):
        self.mean = mean
        self.std = std

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
        self.patient_list = np.unique(np.array(self.patient_list))

    def _load_n_norm_mat(self, file_info: dict, patient_idx: int, mod: str) -> dict:

        #TODO: add documentation and load everything directly in pytorch tensors

        file_name = file_info["dataset"][mod]["filename"]
        data_idx = file_info["data_pos"]
        data_num = file_info["data_num"]
        end_file = file_info["end_file"]
        start_file = file_info["start_file"]
        end_idx = data_num + data_idx

        if "skip_skips" not in self.filter_patients or self.filter_patients["skip_skips"]:
            skip_view = torch.empty(0)
        else:
            skip_view = file_info["skip_views"][mod]

        f = h5py.File(file_name, 'r', swmr=True)
        if "stft" in mod:
            signal = f["X2"][:,:,data_idx:end_idx]
            signal = np.expand_dims(signal, axis=1)
            if self.normalize and hasattr(self,"mean") and hasattr(self,"std"):
                signal = einops.rearrange(signal, "freq channels time inner -> inner time channels freq")
                signal = (signal - self.mean[mod]["ch_0"]) / self.std[mod]["ch_0"]
                signal = einops.rearrange(signal, "inner time channels freq -> inner channels freq time")
        else:
            raise Exception("The current script does not support timeseries that have not being converted to stft, you can add it here!")

        label = f["label"][0,data_idx:end_idx]
        init = torch.zeros(len(label))
        #Return an index init that shows the beginning and the ending of sequential sleep epochs.
        if data_idx == start_file and end_file>data_idx and  len(init) >0:
            init[0] = 1
        elif end_idx == end_file:
            init[-1] = 1

        img = torch.from_numpy(signal).unsqueeze(dim=2)
        label = torch.from_numpy(label).long() -1

        ids = [{ "patient_num":patient_idx,"ids": i} for i in range(data_idx, end_idx)]

        return {"data": img, "label": label, "init": init, "skip_view": skip_view, "ids":ids}

    def _find_file_to_open(self, file_cumul_idx, previous_output) -> dict:

        if "index" not in previous_output: raise ValueError("Missing attribute 'index' in previous_output of self.chech_file in dataset")
        if "remaining_sleep_epochs" not in previous_output: raise ValueError("Missing attribute 'remaining_sleep_epochs' in previous_output of self.chech_file in dataset")

        index = previous_output["index"]
        remaining_sleep_epochs = previous_output["remaining_sleep_epochs"]

        patient_num = -1
        if self.cumulatives["lengths"][file_cumul_idx + 1] > index and self.cumulatives["lengths"][file_cumul_idx] <= index:
            cumul_file = self.cumulatives["files"]["{}-{}".format(self.cumulatives["lengths"][file_cumul_idx], self.cumulatives["lengths"][file_cumul_idx + 1])]
            data_idx = index - self.cumulatives["lengths"][file_cumul_idx] + cumul_file["data_idx"]["start_idx"]
            new_remaining_sleep_epochs = max(remaining_sleep_epochs - (cumul_file["data_idx"]["end_idx"] - data_idx), 0)
            data_num = remaining_sleep_epochs - new_remaining_sleep_epochs

            previous_output["index"] += data_num
            previous_output["remaining_sleep_epochs"] = new_remaining_sleep_epochs
            new_output = {"data_pos": data_idx, "data_num": data_num,
                          "end_file": cumul_file["data_idx"]["end_idx"],
                          "start_file": cumul_file["data_idx"]["start_idx"],
                          "remaining_sleep_epochs": previous_output["remaining_sleep_epochs"],
                          "dataset": cumul_file["dataset"]
                          }
            patient_num = cumul_file["patient_num"]


            if "skip_views" not in cumul_file:
                new_output["skip_views"] = {view: torch.empty(0) for view in self.dataset}
            elif type(cumul_file["skip_views"][list(cumul_file["skip_views"].keys())[0]]) == torch.Tensor and ("skip_skips" in self.filter_patients and not self.filter_patients["skip_skips"]):
                new_output["skip_views"] = {view: cumul_file["skip_views"][view][data_idx-cumul_file['data_idx']['start_idx']:data_idx+data_num-cumul_file['data_idx']['start_idx']] for view in self.dataset}

            if cumul_file["patient_num"] in previous_output:
                previous_output[cumul_file["patient_num"]].append(new_output)
            else:
                previous_output.update({cumul_file["patient_num"]: [new_output]})

        return previous_output, patient_num

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

    def _subsample_patients(self, std_per_indices):
        """
        Subsamples patients based on specified criteria.

        Args:
            std_per_indices (dict): A dictionary containing standard deviations per patient.

        Returns:
            dict: A dictionary mapping patient IDs to tensors indicating which modalities to skip.
        """

        # Check if 'filter_patients' contains 'subsets'
        if "subsets" not in self.filter_patients:
            raise ValueError("'filter_patients' must contain a 'subsets' key.")

        # Check if 'subsets' contains the required modalities
        required_modalities = ["combined", "eeg", "eog"]
        for modality in required_modalities:
            if modality not in self.filter_patients["subsets"]:
                raise ValueError(f"'subsets' must contain '{modality}' key.")

        # Step 1: Sample patients for the combined subset
        combined_set = random.sample(list(self.patient_list), self.filter_patients["subsets"]["combined"])
        skip_patient_ids = {
            patient: torch.cat([
                torch.zeros(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1),
                torch.zeros(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1)
            ], dim=1)
            for patient in combined_set
        }
        not_chosen_patients = [patient for patient in self.patient_list if patient not in combined_set]

        # Step 2: Sample patients for the EEG subset
        eeg_set = random.sample(not_chosen_patients, self.filter_patients["subsets"]["eeg"])
        skips = {
            patient: torch.cat([
                torch.zeros(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1),
                torch.ones(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1)
            ], dim=1)
            for patient in eeg_set
        }
        skip_patient_ids.update(skips)
        not_chosen_patients = [patient for patient in self.patient_list if patient not in combined_set]

        # Step 3: Sample patients for the EOG subset
        eog_set = random.sample(not_chosen_patients, self.filter_patients["subsets"]["eog"])
        skips = {
            patient: torch.cat([
                torch.ones(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1),
                torch.zeros(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1)
            ], dim=1)
            for patient in eog_set
        }
        skip_patient_ids.update(skips)

        # Store the subsampled patients
        self.subsampled_patients = {
            "combined": combined_set,
            "eeg": eeg_set,
            "eog": eog_set,
            "all": combined_set + eeg_set + eog_set
        }

        # Print information about the subsampled patients
        print("We subsample patients with {} both mods, {} only EEG and {} only EOG and len_skipped is {}.".format(
            len(combined_set), len(eeg_set), len(eog_set), len(skip_patient_ids)
        ))

        return skip_patient_ids

    def _get_broken_modalities(self, filename):
        """
        Retrieves information about patients with broken modalities based on standard deviations.

        Args:
            filename (str): Path to the file containing standard deviation information.

        Returns:
            dict: A dictionary mapping patient IDs to tensors indicating which modalities to skip.
        """

        # Load standard deviation data from the specified file
        with open(filename, "rb") as file:
            std_per_indices = pickle.load(file)

        # Check if subsampling is requested
        if self.filter_patients["use_type"] == "subsample":
            return self._subsample_patients(std_per_indices)

        # Check if 'std_threshold' and 'perc_threshold' keys exist in 'filter_patients'
        if "std_threshold" not in self.filter_patients:
            raise ValueError("'filter_patients' must contain a 'std_threshold' key.")
        if "perc_threshold" not in self.filter_patients:
            raise ValueError("'filter_patients' must contain a 'perc_threshold' key.")

        # Retrieve thresholds
        threshold = self.filter_patients["std_threshold"]
        perc_threshold = self.filter_patients["perc_threshold"]

        # Compute differences between EEG and EOG standard deviations
        mod_diff = {
            i: (std_per_indices[i]["std_eeg"][:, 2] - std_per_indices[i]["std_eog"][:, 2]).numpy()
            for i in std_per_indices.keys() if i in self.patient_list
        }

        # Identify patients with broken modalities
        perc_t = {i: (np.abs(mod_diff[i]) > threshold).sum() / len(mod_diff[i]) for i in mod_diff.keys()}
        patients_chosen = np.array([i for i in perc_t if perc_t[i] > perc_threshold])
        print("Patients with broken modalities are {}".format(len(patients_chosen)))

        # Separate patients with broken EEG and EOG modalities
        perc_t = {i: (mod_diff[i] > threshold).sum() / len(mod_diff[i]) for i in mod_diff.keys()}
        patients_chosen_eeg = np.array([i for i in perc_t if perc_t[i] > perc_threshold])
        print("Patients with broken EEG modality are {}".format(len(patients_chosen_eeg)))

        perc_t = {i: (-mod_diff[i] > threshold).sum() / len(mod_diff[i]) for i in mod_diff.keys()}
        patients_chosen_eog = np.array([i for i in perc_t if perc_t[i] > perc_threshold])
        print("Patients with broken EOG modality are {}".format(len(patients_chosen_eog)))

        # Construct skip_patient_ids dictionary
        skip_patient_ids = {}
        for i in patients_chosen:
            skip_mod = torch.zeros(len(mod_diff[i]), 2)
            skip_mod[mod_diff[i] > threshold, 0] = 1
            skip_mod[mod_diff[i] < -threshold, 1] = 1
            skip_patient_ids[i] = skip_mod

        return skip_patient_ids

    def _single_patient_cumulative_includeskipped(self, patient_num: int, file_len: int, file_idx: int) -> None:
        """
        Computes cumulative information for a single patient, considering skipped modalities.

        Args:
            patient_num (int): Patient identifier.
            file_len (int): Total number of files.
            file_idx (int): Index of the current file.
        """
        if patient_num in self.broken_mod_dict:
            # Check if whole patient processing is requested
            if self.filter_patients["whole_patient"]:
                self._single_patient_cumulative_full(patient_num, file_len, file_idx)
                return

            this_broken_patient = copy.deepcopy(self.broken_mod_dict[patient_num])
            # Multiply the EOG column by 2 to distinguish it from EEG
            this_broken_patient[:,1:2] *= 2
            # Compute consecutive skip labels and lengths
            skip_labels, skip_labels_lengths = torch.unique_consecutive(this_broken_patient.sum(dim=1), return_counts=True)
            count, consecutives = 0, []
            for i in range(len(skip_labels)):
                end = count + skip_labels_lengths[i]
                consecutives.append({"start": count,
                                     "end": end,
                                     "skip_label":skip_labels[i], #This contains the type of this segment 0: none missing, 1: eeg missing, 2: eog missing, 3: both missing
                                     "skip_views": {
                                        "stft_eeg": self.broken_mod_dict[patient_num][count:count + skip_labels_lengths[i],:1].squeeze(),
                                        "stft_eog": self.broken_mod_dict[patient_num][count:count + skip_labels_lengths[i],1:].squeeze()}
                                     })
                count = count + skip_labels_lengths[i]

            for cons in consecutives:
                if cons["skip_label"] == 3 or cons["skip_label"] == 0: continue # Dont include parts that have both eeg and eog broken or none
                self.cumulatives["lengths"].append(cons["end"] - cons["start"] + self.cumulatives["lengths"][-1])
                self.cumulatives["files"]["{}-{}".format(self.cumulatives["lengths"][-2], self.cumulatives["lengths"][-1])] = \
                    {
                        "patient_num": patient_num,
                        "data_idx": {"start_idx": cons["start"], "end_idx": cons["end"]},
                        "dataset": {view: self.dataset[view]["dataset"][file_idx] for view in self.dataset},
                        "skip_views": cons["skip_views"]
                    }

    def _single_patient_cumulative_subsample(self, patient_num: int, file_len: int, file_idx: int) -> None:
        self.cumulatives["lengths"].append(file_len + self.cumulatives["lengths"][-1])
        start_idx = 0
        end_idx = file_len
        self.cumulatives["files"]["{}-{}".format(self.cumulatives["lengths"][-2], self.cumulatives["lengths"][-1])] = {
            "patient_num": patient_num,
            "data_idx": {"start_idx": start_idx, "end_idx": end_idx},
            "dataset": {view: self.dataset[view]["dataset"][file_idx] for view in self.dataset}
        }

    def _single_patient_cumulative_full(self, patient_num: int, file_len: int, file_idx: int):
        """
        This method gathers the file information and adds it to the self.cumulatives on the correct key based on its length and the length  of the previous ones
        Example:
             self.cumulatives["lengths"] is  [0,1081, 2085] and
             self.cumulatives["files"] is {'0-1081': {'patient_num': 1, 'dataset': ..},
                                           '1081-2085': {'patient_num': 2, 'dataset': ..}}

             a new consecutive for patient 3 is {"start": 10, "end": 510, "skip_views":..}
             then cumulatives transforms to:
             self.cumulatives["lengths"] is  [0,1081, 2085, 2585] and
             self.cumulatives["files"] is {'0-1081': {'patient_num': 1, 'dataset': ..},
                                           '1081-2085': {'patient_num': 2, 'dataset': ..}
                                           '2085-2585': {'patient_num': 3, 'dataset': ..}}

            We can have many consecutives essentially referring to non-interruptive labeled sleep time from the same patient
            we would treat each such consecutive as if it was coming from another patient.
        """
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

    def print_statistics_per_patient(self):
        """
        Process the dataset, analyzing labels for each file.

        Prints information about the number of windows with different labels in each file.
        """
        for i in range(0, len(self.dataset), 2):
            for patient in range(len(self.dataset[i])):
                if "empty" in self.dataset[i][patient]:
                    logging.info("File: empty")
                    continue

                # Load the HDF5 file
                with h5py.File(self.dataset[i][patient], 'r') as f:
                    labels = np.array(f["labels"]).argmax(axis=1)
                    unique_labels, label_counts = np.unique(labels, return_counts=True)

                    # Construct a summary string
                    summary = f"File: {self.dataset[i][patient]} has {len(labels)} windows with labels "
                    for label, count in zip(unique_labels, label_counts):
                        summary += f"{label}-{count:04} "

                    logging.info(summary)

    def _load_view_files(self, file_output: dict, view: str) -> dict:

        view_output = defaultdict(lambda: [])
        for patient_idx, this_patient_instr in file_output.items():
            for seqs in this_patient_instr:
                loaded_segments = self._load_n_norm_mat(file_info=seqs,
                                                     patient_idx=patient_idx,
                                                     mod=view)
                for i in loaded_segments:
                    view_output[i].append(loaded_segments[i])

        return dict(view_output)

    def _aggegate_n_update_view(self, output: nested_dict, view_output: dict, view: str):

        # Aggregate results for each file_output you have opened.
        for out_i in view_output:
            if out_i == "ids":
                view_output["ids"] = [item for sublist in view_output["ids"] for item in sublist]  # flatten the list
                # the following lines are aggregating the ids and the patient num to the output
                ids = torch.cat([torch.Tensor([int(id["ids"])]) for id in view_output["ids"]])
                patient_nums = torch.cat([torch.Tensor([int(id["patient_num"])]) for id in view_output["ids"]])
                total_ids = torch.cat([patient_nums.unsqueeze(dim=1), ids.unsqueeze(dim=1)], dim=1)
                output[out_i].update({view: total_ids})
            else:
                # print([i.shape for i in view_output[out_i]])
                output[out_i].update({view: torch.cat(view_output[out_i])})
    def __getitem__(self, index):

        """

        file_output is organised as
        {patient_idx:{
              "dataset":{"view_name": list_of_files, "second_view_name":list_of_files} # each file is a dict with "filename" and "len_windows" how many windows the file contains
              "data_pos": int, # which position in the dataset we should get the data from
              "data_num": int, # how many should we get from this position
              "remaining": int, # how many remain to fill the outer_sequence
              },
              {"dataset":list_of_files},
              }

        """

        index = index * self.outer_seq_length # cast it from batch to epoch space number.

        file_output = {"remaining_sleep_epochs": self.outer_seq_length, "index": index}
        #find the files that correspond to the given index!
        for file_cumul_idx in range(len(self.cumulatives["lengths"]) - 1):
            #check files and cumulative to find which one we have to open and which parts of the file to use.
            file_output, last_pat_num = self._find_file_to_open(file_cumul_idx=file_cumul_idx, previous_output=file_output)
            #Iterate over the next one till sequential sleep epochs are filled and this batch is completed.
            if (last_pat_num in file_output) and file_output[last_pat_num][-1]["remaining_sleep_epochs"]==0: break
        #remove useless_info
        file_output.pop("index")
        file_output.pop("remaining_sleep_epochs")

        output = nested_dict()
        for view in self.views: # stft_eeg, stft_eog, time_eeg, time_eog etc.
            view_output = self._load_view_files(file_output=file_output, view=view)
            self._aggegate_n_update_view(output=output, view_output=view_output, view=view)
        output["idx"] = output["ids"]["stft_eeg"] if "stft_eeg" in output["ids"] else output["ids"]["stft_eog"]
        output["label"] = output["label"]["stft_eeg"] if "stft_eeg" in output["label"] else output["label"]["stft_eog"]
        # print(output["data"]["stft_eeg"].shape)
        # {"data": {0: spectrogram, 1: images, 2: audio, 3: face_features, 4: face_image}, "label": label, "idx": idx}
        return output

    def __len__(self):
        return int(self.cumulatives["lengths"][-1]/self.outer_seq_length)

class SleepDataLoader():

    def __init__(self, config: easydict.EasyDict):
        """
        :param config:
        """

        self.config = config

        sleep_dataset_train, sleep_dataset_val, sleep_dataset_test, sleep_dataset_total = self._get_datasets()

        g = torch.Generator()
        g.manual_seed(0)

        num_cores = len(os.sched_getaffinity(0))-1
        num_cores = 0
        print("Available cores {}".format(len(os.sched_getaffinity(0))))
        print("We are changing dataloader workers to num of cores {}".format(num_cores))

        self.train_loader = torch.utils.data.DataLoader(sleep_dataset_train,
                                                        batch_size=self.config.training_params.batch_size,
                                                        num_workers=num_cores,
                                                        pin_memory=self.config.training_params.pin_memory,
                                                        generator=g,
                                                        worker_init_fn=lambda worker_id: np.random.seed(15 + worker_id))
        self.valid_loader = torch.utils.data.DataLoader(sleep_dataset_val,
                                                        batch_size=self.config.training_params.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=self.config.training_params.data_loader_workers,
                                                        pin_memory=self.config.training_params.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(sleep_dataset_test,
                                                       batch_size=self.config.training_params.test_batch_size,
                                                       shuffle=False,
                                                       num_workers=self.config.training_params.data_loader_workers,
                                                       pin_memory=self.config.training_params.pin_memory)

        self.total_loader = torch.utils.data.DataLoader(sleep_dataset_total,
                                                        batch_size=self.config.training_params.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=self.config.training_params.data_loader_workers,
                                                        pin_memory=self.config.training_params.pin_memory)

        self.norm_agent = Normalization_finder(dataloader = self, config=config)
        self.metrics = self.norm_agent.get_norm_metrics()

        logging.info("Train: {}, Val: {}, Test: {}".format(len(self.train_loader),len(self.valid_loader), len(self.test_loader)))

        if self.config.statistics.print:
            self._statistics_mat()

    def _get_datasets(self):

        views = {}
        for i in self.config.dataset.data_view_dir:
            i["list_dir"] = pjoin(self.config.dataset.data_roots, i["list_dir"])
            views[i["data_type"] + "_" + i["mod"]] = i
        views = self._read_dirs_mat(views)
        train_views, val_views, test_views = self._split_data_mat(views)

        train_dataset = Sleep_Dataset(config=self.config, views=train_views, set_name="train")
        valid_dataset = Sleep_Dataset(config=self.config, views=val_views, set_name="val")
        test_dataset = Sleep_Dataset(config=self.config, views=test_views, set_name="test")
        total_dataset = Sleep_Dataset(config=self.config, views=views, set_name="total")

        return train_dataset, valid_dataset, test_dataset, total_dataset

    def _read_dirs_mat(self, view_dirs: list):
        """
        Read view_dirs the list of dicts that contain the txt file that has all the paths to the datapoints, the type of the data, modality's name and number of channels

        Example:
            view_dirs = [{"list_dir": "patient_mat_list.txt", "data_type": "stft", "mod":  "eeg", "num_ch": 1}]

        Returns the view_dirs list with each dict enhanced with the list of dicts dataset= [{"filename": "patient.npy", "len_windows": 100}], len_windows are the number of sleep epochs the file contains
        """

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

    def _unique_dataloader(self, dataset: dict) -> (torch.Tensor, dict):

        """
        This method opens the h5py files of the dataset and returns the label distribution of the dataset
        returns:
               total_classes, counts
        """

        counts = {}
        for view in dataset:
            counts[view] = torch.zeros(self.config.model.args.num_classes)

        total_classes = torch.arange(self.config.model.args.num_classes)
        for view in dataset:
            for file in tqdm(dataset[view]["dataset"], desc="Counting", total=len(dataset[view]["dataset"])):
                if "empty" in file["filename"]:
                    continue

                # Load labels from HDF5 file
                with h5py.File(file["filename"], "r") as f:
                    labels = torch.tensor(np.array(f["label"])).squeeze()

                    if len(labels.shape) > 1 and labels.shape[1] == 2:
                        labels = labels[:, 0].squeeze()

                    labels = labels - 1 # Labels start from 1, so we subtract 1 to start from 0

                    if len(labels.shape) > 1:
                        labels = labels.argmax(axis=1)

                    unique_labels, label_counts = torch.unique(labels, return_counts=True)

                    for i, cl in enumerate(unique_labels):
                        counts[view][int(cl)] += label_counts[int(i)]

        return total_classes, counts

    def _statistics_mat(self):

        def __compute_dataset_stats(dataset: dict, set_name: str) -> list:
            classes, counts = self._unique_dataloader(dataset)
            return [set_name, classes, counts]

        total_stats = __compute_dataset_stats(dataset=self.total_loader.dataset.dataset, set_name="Total")
        train_stats = __compute_dataset_stats(dataset=self.train_loader.dataset.dataset, set_name="Training")
        valid_stats = __compute_dataset_stats(dataset=self.valid_loader.dataset.dataset, set_name="Validation")
        test_stats = __compute_dataset_stats(dataset=self.test_loader.dataset.dataset, set_name="Test")

        all_stats = [total_stats, train_stats, valid_stats, test_stats]

        class_dict = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

        if self.config.statistics["print"]:
            for label, classes, counts in all_stats:
                logging.info(f"In {label} set, we have:")
                for view in counts:
                    view_summary = ", ".join(f" {class_dict[cl.item()]}: {int(count)}" for cl, count in zip(classes, counts[view]))
                    logging.info(f"{view}: {view_summary}")

    def _split_data_mat(self, views: list) -> (list, list, list):

        if (self.config.dataset.data_split.split_method == "patients_test"):
            dirs_train, dirs_val, dirs_test = self._split_patients_test(dirs_train_whole=views,
                                                                        split_rate_val=self.config.dataset.data_split.val_split_rate,
                                                                        split_rate_test=self.config.dataset.data_split.test_split_rate )
        elif (self.config.dataset.data_split.split_method == "patients_sleeptransformer"):
            logging.info("We are splitting dataset by SleepTransformer split")
            dirs_train, dirs_val, dirs_test = self._split_patients_sleeptf(dirs_train_whole=views)
        else:
            raise ValueError("No splitting method named {} exists.".format(self.config.dataset.data_split.split_method))
        return dirs_train, dirs_val, dirs_test

    def _split_patients_test(self, dirs_train_whole:list, split_rate_val:float, split_rate_test:float):

        """
        This method split patients among the different train/val/test without any patient participating in more than one. It doesnt take into account the distribution of the labels.

        Returns:
             train_views, val_views, test_views : list of dicts for the corresponding patients derived dirs_train_whole
        """
        view = list(dirs_train_whole.keys())[0]
        # patient_names = np.array([i["filename"].split("/")[-1][:5] for i in dirs_train_whole[view]["dataset"]])

        # splits = {}
        # for i in range(10):
        #     np.random.seed(i)
        #     np.random.shuffle(patient_names)
        #     trainval_test = train_test_split(patient_names, test_size=split_rate_test, random_state=self.config.training_params.seed)
        #     train_val = train_test_split(trainval_test[0], test_size=split_rate_val, random_state=self.config.training_params.seed)
        #
        #
        #     splits[i] = {"train": train_val[0], "val": train_val[1], "test": trainval_test[1]}
        # #save the splits in pkl file
        # with open("./datasets/SHHS/trainvaltest_splits.pkl", "wb") as f:
        #     pickle.dump(splits, f)

        #read teh splits from pkl file
        with open("./datasets/trainvaltest_splits.pkl", "rb") as f:
            splits = pickle.load(f)

        train_views = copy.deepcopy(dirs_train_whole)
        val_views = copy.deepcopy(dirs_train_whole)
        test_views = copy.deepcopy(dirs_train_whole)


        this_split = splits[self.config.dataset.fold]


        for view in dirs_train_whole:
            train_dataset, val_dataset, test_dataset = [], [], []
            for file in dirs_train_whole[view]["dataset"]:
                patient_num = file["filename"].split("/")[-1][:5]
                if patient_num in this_split["test"]:
                    test_dataset.append(file)
                elif patient_num in this_split["train"]:
                    train_dataset.append(file)
                elif patient_num in this_split["val"]:
                    val_dataset.append(file)
                else:
                    raise Warning("Splitting is not going well, some patient has no house. \n {}".format(file))

            train_views[view]["dataset"] = train_dataset
            val_views[view]["dataset"] = val_dataset
            test_views[view]["dataset"] = test_dataset

            logging.info("{} Train {} validation {} test {}".format(view, len(train_views[view]["dataset"]), len(val_views[view]["dataset"]), len(test_views[view]["dataset"])))

        return train_views, val_views, test_views

        # return np.array(dirs_train_whole[:,split_dict["train"]]), np.array(dirs_train_whole[:,split_dict["valid"]]), np.array(dirs_train_whole[:,split_dict["test"]])

    def _split_patients_sleeptf(self, dirs_train_whole:list):
        # patient_names = np.unique(np.array([i.split("/")[-1][:5] for i in dirs_train_whole[0]]))
        # # cross_val_run = pickle.load(open(self.config.folds_file, "rb"))
        f = loadmat(self.config.dataset.data_split.folds_file)
        f["train_sub"] = f["train_sub"].squeeze() - 1
        f["eval_sub"] = f["eval_sub"].squeeze() - 1
        f["test_sub"] = f["test_sub"].squeeze() - 1

        train_views = copy.deepcopy(dirs_train_whole)
        val_views = copy.deepcopy(dirs_train_whole)
        test_views = copy.deepcopy(dirs_train_whole)

        #We need a running difference between the numbers of SleepTF patients and ours. When we skip a patient we skip a number, SleepTF doesnt.
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

            train_views[view]["dataset"] = train_dataset
            val_views[view]["dataset"] = val_dataset
            test_views[view]["dataset"] = test_dataset

        return train_views, val_views, test_views

class Normalization_finder():
    def __init__(self, dataloader, config):
        self.config = config
        self.dataloader = dataloader

    def load_metrics(self):

        logging.info("Loading metrics from {}".format(self.config.dataset.norm_dir)) #path to pkl file
        metrics_file = open(self.config.dataset.norm_dir, "rb")
        self.metrics = pickle.load(metrics_file)

        # mean, std = {}, {}
        # if self.config.normalization.metrics_type == "per_recording":
        #     mean = self.metrics["mean"]
        #     std = self.metrics["std"]
        # if self.config.normalization.metrics_type == "sep_total_train_test":
        #     mean = self.metrics["mean"]
        #     std = self.metrics["std"]
        # elif self.config.normalization.metrics_type == "total_dataset":
        #     for i, f in enumerate(self.config.data_view_dir):
        #         mod = f["data_type"] + "_" + f["mod"]
        #         mean[mod] = self.metrics["mean"][mod]
        #         std[mod] = self.metrics["std"][mod]
        # elif self.config.normalization.metrics_type == "train_dataset":
        #     for i, f in enumerate(self.config.dataset.data_view_dir):
        #         mod = f["data_type"] + "_" + f["mod"]
        #         if "time" in mod:
        #             mean[mod] = 0
        #             std[mod] = 1
        #         else:
        #             mean[mod] = self.metrics["mean"][mod]
        #             std[mod] = self.metrics["std"][mod]
        # else:
        #     logging.error("Unknown metric type check config.normalization.metric_type")
        #     raise ValueError("Unknown metric type check config.normalization.metric_type")

    def get_norm_metrics(self):

        self.load_metrics()  # Load metrics from file
        self.dataloader.train_loader.dataset.set_mean_std(self.metrics["mean"], self.metrics["std"])
        self.dataloader.valid_loader.dataset.set_mean_std(self.metrics["mean"], self.metrics["std"])
        self.dataloader.total_loader.dataset.set_mean_std(self.metrics["mean"], self.metrics["std"])
        self.dataloader.test_loader.dataset.set_mean_std(self.metrics["mean"], self.metrics["std"])

        return self.metrics

    def load_metrics_ongoing(self, metrics):
        mean = metrics["mean"]
        std = metrics["std"]
        self.metrics = metrics
        self.dataloader.train_loader.dataset.set_mean_std(mean, std)
        self.dataloader.valid_loader.dataset.set_mean_std(mean, std)
        # self.test_loader.dataset.set_mean_std(mean, std)
        self.dataloader.total_loader.dataset.set_mean_std(mean, std)

        if self.config.normalization.metrics_type == "sep_total_train_test":
            if "mean_test" not in metrics or "std_test" not in metrics:
                mean_test, std_test = self.calculate_mean_std_mymat_total(self.test_loader.dataset.dataset)
                self.test_loader.dataset.set_mean_std(mean_test, std_test)
            else:
                self.test_loader.dataset.set_mean_std(metrics["mean_test"], metrics["std_test"])
        else:
            self.dataloader.test_loader.dataset.set_mean_std(mean, std)

