from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import ToTensor
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

class Sleep_Dataset(Dataset):

    def __init__(self, file_dirs, num_views, seq_length, seq_views, keep_view, inner_overlap, data_augmentation = {} ):
        super()
        self.dataset = file_dirs
        self.num_views = num_views
        self.seq_views = seq_views
        self.keep_view = keep_view
        self.inner_overlap = inner_overlap
        self.data_augm_times = len(data_augmentation.keys()) - 2
        self.augmentation_type = data_augmentation['type'] if 'type' in data_augmentation.keys() else 'same'
        self.augmentation_rate = data_augmentation['rate'] if 'rate' in data_augmentation.keys() else 0

        self.aug = data_augmentation
        self.tf = Transform_Images(data_augmentation)

        #This assert is used in case we want the label from the center one (seq to one)
        # assert seq_length[0] >0 and seq_length[0]%2 !=0, "Outer sequence length must be a positive odd integer"
        self.outer_seq_length = seq_length[0]
        self.inner_seq_length = seq_length[1]
        self._get_len()

    def _get_len(self):

        self.dataset_true_length = int(len(self.dataset[0])/self.outer_seq_length) if not self.dataset.shape[0] < 1 else 0

        if self.augmentation_type == "mul":
            self.dataset_aug_length = int(self.dataset_true_length * self.data_augm_times + 1)
        elif self.augmentation_type == "same":
            self.dataset_aug_length = self.dataset_true_length

    def __getitem__(self, index):
        data_idx = index % self.dataset_true_length
        if self.augmentation_type == "mul":
            augment_number = index // self.dataset_true_length
        elif self.augmentation_type == "same":
            same = np.random.choice([0,1], p=[1 - self.augmentation_rate, self.augmentation_rate])
            if same == 1 :
                same = np.random.randint(1,self.data_augm_times+1)
            augment_number = same


        filenames = []
        for view_i in range(0, 3*self.num_views, 3):
            filenames.append(self.dataset[view_i][data_idx*self.outer_seq_length:data_idx*self.outer_seq_length+self.outer_seq_length])
            if view_i == 0:
                label = self.dataset[view_i+1][data_idx*self.outer_seq_length:data_idx*self.outer_seq_length+self.outer_seq_length]
                init = self.dataset[view_i+2][data_idx*self.outer_seq_length:data_idx*self.outer_seq_length+self.outer_seq_length]
            else:
                sub_label = self.dataset[view_i+1][data_idx*self.outer_seq_length:data_idx*self.outer_seq_length+self.outer_seq_length]
                sub_init = self.dataset[view_i+2][data_idx*self.outer_seq_length:data_idx*self.outer_seq_length+self.outer_seq_length]
                if not ((sub_label == label).all() or (sub_init == init).all()):
                    raise ValueError("Labels between the two views are not the same")

        # print("{}_{}".format(self.dataset_aug_length, filenames[0][0]))
        images = []
        for seq_files in filenames:
            images_local = []
            for file in seq_files:
                if file.split(".")[-1] =="png" or file.split(".")[-1] =="jpg":
                    images_local.append(ToTensor()(Image.open(file)).unsqueeze(dim=0))
                elif file.split(".")[-1] =="npz":
                    if self.outer_seq_length > 1:
                        images_local.append(torch.from_numpy(np.load(file,allow_pickle=True)["arr_0"]).unsqueeze(dim=0).unsqueeze(dim=0))
                    elif self.outer_seq_length == 1:
                        images_local.append(torch.from_numpy(np.load(file,allow_pickle=True)["arr_0"]).unsqueeze(dim=0))
                    elif self.inner_seq_length >0:
                        images_local.append(torch.from_numpy(np.load(file,allow_pickle=True)["arr_0"]).unsqueeze(dim=0))
            images_local = torch.cat(images_local)
            images_local[images_local != images_local] = -20.0
            images.append(images_local)


        #This is a convention to make view_1 shape channel=1 * height * width
        # images = [i.unsqueeze(dim=0) for i in images]
        label = torch.from_numpy(np.array([int(i) for i in label]))
        init = torch.from_numpy(np.array([int(i) for i in init]))
        ids = torch.arange(data_idx*self.outer_seq_length,data_idx*self.outer_seq_length+self.outer_seq_length)


        if augment_number > 0 : images = self.transform_images(images, augment_number)

        output = copy.deepcopy(images)
        #Reshape img to create the inner windows
        if self.inner_seq_length != 0:
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

        return output, label, init,  ids

    def __len__(self):
        return self.dataset_aug_length

    def choose_specific_patient(self, patient_num):
        changed_dirs = []
        for i in range(len(self.dataset)): changed_dirs.append([])
        for i in range(len(self.dataset[0])):
            if  "patient_{}".format(f'{patient_num:02}') in self.dataset[0][i]:
                for j in range(len(self.dataset)):
                    changed_dirs[j].append(self.dataset[j][i])
        self.dataset = np.array(changed_dirs)
        self._get_len()

    def transform_images(self, images, num):
        aug_method = getattr(self.tf, self.aug[str(num)]["method"])
        # aug_method = globals()[self.aug[num]["method"]]
        for i in range(len(images)):
            # print("{}_{}".format(i,num))
            images[i] = aug_method(images[i], self.aug[str(num)], num)
        return images

class SleepDataLoader():

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        sleep_dataset_train, sleep_dataset_val, sleep_dataset_test = self._get_datasets()

        shuffle_training_data =  False if self.config.seq_legth[0]>1 else True
        shuffle_training_data = True

        self.train_loader = torch.utils.data.DataLoader(sleep_dataset_train, batch_size=self.config.batch_size,
                                                        shuffle=shuffle_training_data, num_workers=self.config.data_loader_workers,
                                                        pin_memory=self.config.pin_memory,
                                                        worker_init_fn=_init_fn)
        self.valid_loader = torch.utils.data.DataLoader(sleep_dataset_val, batch_size=self.config.test_batch_size,
                                                        shuffle=False, num_workers=self.config.data_loader_workers,
                                                        pin_memory=self.config.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(sleep_dataset_test, batch_size=self.config.test_batch_size,
                                                       shuffle=False, num_workers=self.config.data_loader_workers,
                                                       pin_memory=self.config.pin_memory)

        self._statistics()


    def _get_datasets(self):
        views_train = [self.config.data_roots + "/" + i[0] for i in self.config.data_view_dir]
        views_test = [self.config.data_roots + "/" + i[1] for i in self.config.data_view_dir]

        dirs_train_whole, train_len = self._read_dirs(views_train)
        dirs_test, test_len = self._read_dirs(views_test)
        dirs_train, dirs_val = self._split_data(dirs_train_whole)
        dirs_train = dirs_train.sort()
        if not hasattr(self.config,"seq_legth"):
            self.config.seq_legth = [1,0]

        return Sleep_Dataset(dirs_train, train_len, self.config.seq_legth, self.config.seq_views, self.config.keep_view, self.config.inner_overlap, self.config.augmentation), \
               Sleep_Dataset(dirs_val, train_len, self.config.seq_legth, self.config.seq_views, self.config.keep_view, self.config.inner_overlap), \
               Sleep_Dataset(dirs_test, test_len, self.config.seq_legth, self.config.seq_views, self.config.keep_view, self.config.inner_overlap)

    def _read_dirs(self, view_dirs):
        dataset = []
        for i, view_dir in enumerate(view_dirs):
            with open(view_dir) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\n')
                datafile_names = []
                labels, init = [], []
                for row in csv_reader:
                    dt = row[0].split("-")
                    label = dt[1].split("_")
                    datafile_names.append(dt[0])
                    labels.append(int(label[0]))
                    init.append(int(label[1]))
            dataset.append(datafile_names)
            dataset.append(labels)
            dataset.append(init)
        num_views = len(view_dirs)
        return np.array(dataset), num_views

    def _statistics(self):
        #Calculate weights for unbalanced classes
        classes, counts = np.unique(self.train_loader.dataset.dataset[1,:],return_counts=True)
        val_classes, val_counts = np.unique(self.valid_loader.dataset.dataset[1,:],return_counts=True)
        test_classes, test_counts = np.unique(self.test_loader.dataset.dataset[1,:],return_counts=True)

        print("In training set we got {} images of NQS and {} of QS".format(counts[0],counts[1]))
        print("In validation set we got {} images of NQS and {} of QS".format(val_counts[0],val_counts[1]))
        print("In test set we got {} images of NQS and {} of QS".format(test_counts[0],test_counts[1]))

        self.weights = len(self.train_loader.dataset.dataset[0])/counts
        norm = np.linalg.norm(self.weights)
        self.weights = self.weights / norm
        print(self.weights)

    def _split_data(self, dirs_train_whole):
        if(self.config.split_method == "files"):
            dirs_train, dirs_val =self._split_files(dirs_train_whole)
        elif(self.config.split_method == "patients"):
            dirs_train, dirs_val =self._split_patients(dirs_train_whole)
        elif(self.config.split_method == "patients_num"):
            dirs_train, dirs_val =self._split_patients_num(dirs_train_whole, self.config.val_patient_num)
        elif(self.config.split_method == "random"):
            dirs_train, dirs_val = self._split_random(dirs_train_whole)
        elif(self.config.split_method == "random_folds"):
            dirs_train, dirs_val = self._random_folds(dirs_train_whole, self.config.rand_split ,self.config.rand_splits, self.config.seed)
        else:
            raise ValueError("No splitting method named {} exists.".format(self.config.split_method))
        return dirs_train, dirs_val, dirs_test

    def _split_files(self, dirs_train_whole):
        file_names = np.unique(np.array([i.split("/")[-2] for i in dirs_train_whole[0]]))
        split_names = train_test_split(file_names, test_size=self.config.split_rate, random_state=self.config.seed)
        train_idx, val_idx = [], []
        for index, file_name in enumerate(dirs_train_whole[0]):
            if file_name.split("/")[-2] in split_names[0]:
                train_idx.append(index)
            else:
                val_idx.append(index)
        return np.array(dirs_train_whole[:, train_idx]), np.array(dirs_train_whole[:, val_idx])

    def _split_patients(self, dirs_train_whole):
        patient_names = np.unique(np.array([i.split("/")[-3] for i in dirs_train_whole[0]]))
        split_names = train_test_split(patient_names, test_size=self.config.split_rate, random_state=self.config.seed)
        train_idx, val_idx = [],[]
        for index, file_name in enumerate(dirs_train_whole[0]):
            if file_name.split("/")[-3] in split_names[0]:
                train_idx.append(index)
            else:
                val_idx.append(index)
        return np.array(dirs_train_whole[:,train_idx]), np.array(dirs_train_whole[:,val_idx])

    def _split_patients_num(self, dirs_train_whole, patients_num):
        train_idx, val_idx = [],[]
        if type(patients_num) is list:
            patients_list = ["patient_{}".format(f'{p:02}') for p in patients_num]
        else:
            patients_list = ["patient_{}".format(f'{patients_num:02}')]
        for index, file_name in enumerate(dirs_train_whole[0]):
            if file_name.split("/")[-3] in patients_list:
                val_idx.append(index)
            else:
                train_idx.append(index)
        return np.array(dirs_train_whole[:,train_idx]), np.array(dirs_train_whole[:,val_idx])

    def _random_folds(self, dirs_train_whole, chosen_split, rand_splits, seed):
        skf = StratifiedKFold(n_splits=rand_splits, shuffle=True,random_state=seed)
        skf.get_n_splits(range(len(dirs_train_whole[0])), dirs_train_whole[1])
        for ind,(train_idx, val_idx) in enumerate(skf.split(range(len(dirs_train_whole[0])), dirs_train_whole[1])):
            if ind == chosen_split:
                break
        return np.array(dirs_train_whole[:, train_idx]), np.array(dirs_train_whole[:, val_idx])

    def _split_random(self, dirs_train_whole):
        split_indices = train_test_split(range(len(dirs_train_whole[0])), test_size=self.config.split_rate, random_state=self.config.seed, stratify=dirs_train_whole[1])
        return np.array(dirs_train_whole[:,split_indices[0]]), np.array(dirs_train_whole[:,split_indices[1]])

class SleepDataLoader_EDF():

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        sleep_dataset_train, sleep_dataset_val, sleep_dataset_test = self._get_datasets()

        # shuffle_training_data =  False if self.config.seq_legth[0]>1 elif hasattr(self.config,"shuffle_train") self.config.shuffle_train else True

        if self.config.seq_legth[0]>1: shuffle_training_data=False
        elif  hasattr(self.config,"shuffle_train"): shuffle_training_data=self.config.shuffle_train
        else: shuffle_training_data=True

        self.train_loader = torch.utils.data.DataLoader(sleep_dataset_train, batch_size=self.config.batch_size,
                                                        shuffle=shuffle_training_data, num_workers=self.config.data_loader_workers,
                                                        pin_memory=self.config.pin_memory,
                                                        worker_init_fn=_init_fn)
        self.valid_loader = torch.utils.data.DataLoader(sleep_dataset_val, batch_size=self.config.test_batch_size,
                                                        shuffle=False, num_workers=self.config.data_loader_workers,
                                                        pin_memory=self.config.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(sleep_dataset_test, batch_size=self.config.test_batch_size,
                                                       shuffle=False, num_workers=self.config.data_loader_workers,
                                                       pin_memory=self.config.pin_memory)
        if self.config.print_statistics:
            self._statistics()
        else:
            self.weights = np.ones(self.config.num_classes)/np.ones(self.config.num_classes).sum()
            print(self.weights)

    def _get_datasets(self):
        views_train = [self.config.data_roots + "/" + i[0] for i in self.config.data_view_dir]
        # if self.config.use_test_set:
        #     views_test = [self.config.data_roots + "/" + i[1] for i in self.config.data_view_dir]
        #     dirs_test, _ = self._read_dirs(views_test)
        # else:
        #     dirs_test = np.array([])
        dirs_train_whole, train_len = self._read_dirs(views_train)
        #This code can be used to generate a test set from the training data
        #dirs_train_whole, dirs_test = self._split_data(dirs_train_whole, self.config.test_split_rate)
        dirs_train, dirs_val, dirs_test = self._split_data(dirs_train_whole, self.config.val_split_rate)
        if not hasattr(self.config,"seq_legth"):
            self.config.seq_legth = [1,0]

        return Sleep_Dataset(dirs_train, train_len, self.config.seq_legth, self.config.seq_views, self.config.keep_view, self.config.inner_overlap, self.config.augmentation), \
               Sleep_Dataset(dirs_val, train_len, self.config.seq_legth, self.config.seq_views, self.config.keep_view, self.config.inner_overlap), \
               Sleep_Dataset(dirs_test, train_len, self.config.seq_legth, self.config.seq_views, self.config.keep_view, self.config.inner_overlap)

    def _read_dirs(self, view_dirs):
        dataset = []
        for i, view_dir in enumerate(view_dirs):
            with open(view_dir) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\n')
                datafile_names = []
                labels, init = [], []
                for row in csv_reader:
                    dt = row[0].split("-")
                    label = dt[1].split("_")
                    datafile_names.append(dt[0])
                    labels.append(int(label[0]))
                    init.append(int(label[1]))
            dataset.append(datafile_names)
            dataset.append(labels)
            dataset.append(init)
        num_views = len(view_dirs)
        return np.array(dataset), num_views

    def _statistics(self):
        total = []
        #Calculate weights for unbalanced classes
        classes, counts = np.unique(self.train_loader.dataset.dataset[1,:],return_counts=True)
        total.append(["training", classes, counts])
        val_classes, val_counts = np.unique(self.valid_loader.dataset.dataset[1,:],return_counts=True)
        total.append(["valdiation", val_classes, val_counts])
        if self.config.use_test_set:
            test_classes, test_counts = np.unique(self.test_loader.dataset.dataset[1,:],return_counts=True)
            total.append(["test", test_classes, test_counts])

        for label, cl, c in total:
            s = "In {} we got ".format(label)
            for i in range(len(counts)):
                s = s + "Label {} : {} ".format(cl[i], c[i])
            print(s)
        # print("In training set we got {} images of NQS and {} of QS".format(counts[0],counts[1]))
        # print("In validation set we got {} images of NQS and {} of QS".format(val_counts[0],val_counts[1]))
        # print("In test set we got {} images of NQS and {} of QS".format(test_counts[0],test_counts[1]))

        self.weights = len(self.train_loader.dataset.dataset[0])/counts
        norm = np.linalg.norm(self.weights)
        self.weights = self.weights / norm
        print(self.weights)

    def _split_data(self, dirs_train_whole, split_rate):
        dirs_test = np.array([])
        if(self.config.split_method == "files"):
            dirs_train, dirs_val =self._split_files(dirs_train_whole, split_rate)
        elif(self.config.split_method == "patients"):
            dirs_train, dirs_val =self._split_patients(dirs_train_whole, split_rate)
        elif (self.config.split_method == "patients_test"):
            dirs_train, dirs_val, dirs_test = self._split_patients_test(dirs_train_whole, self.config.val_split_rate, self.config.test_split_rate )
        elif(self.config.split_method == "patients_num"):
            dirs_train, dirs_val =self._split_patients_num(dirs_train_whole, self.config.val_patient_num)
        elif(self.config.split_method == "random"):
            dirs_train, dirs_val = self._split_random(dirs_train_whole, split_rate)
        elif(self.config.split_method == "random_folds"):
            dirs_train, dirs_val = self._random_folds(dirs_train_whole, self.config.rand_split ,self.config.rand_splits, self.config.seed)
        elif(self.config.split_method == "patients_folds"):
            dirs_train, dirs_val = self._split_patients_folds(dirs_train_whole, self.config.fold, self.config.fold_size)
        elif(self.config.split_method == "huy_folds"):
            dirs_train, dirs_val, dirs_test  = self._split_huy_folds(dirs_train_whole, self.config.fold, self.config.fold_size)
        else:
            raise ValueError("No splitting method named {} exists.".format(self.config.split_method))
        return dirs_train, dirs_val, dirs_test

    def _split_files(self, dirs_train_whole, split_rate):
        file_names = np.unique(np.array([i.split("/")[-2] for i in dirs_train_whole[0]]))
        split_names = train_test_split(file_names, test_size=split_rate, random_state=self.config.seed)
        train_idx, val_idx = [], []
        for index, file_name in enumerate(dirs_train_whole[0]):
            if file_name.split("/")[-2] in split_names[0]:
                train_idx.append(index)
            else:
                val_idx.append(index)
        return np.array(dirs_train_whole[:, train_idx]), np.array(dirs_train_whole[:, val_idx])

    def _split_patients(self, dirs_train_whole, split_rate):
        patient_names = np.unique(np.array([i.split("/")[-3] for i in dirs_train_whole[0]]))
        split_names = train_test_split(patient_names, test_size=split_rate, random_state=self.config.seed)
        train_idx, val_idx = [],[]
        for index, file_name in enumerate(dirs_train_whole[0]):
            if file_name.split("/")[-3] in split_names[0]:
                train_idx.append(index)
            else:
                val_idx.append(index)
        return np.array(dirs_train_whole[:,train_idx]), np.array(dirs_train_whole[:,val_idx])

    def _split_patients_test(self, dirs_train_whole, split_rate_val, split_rate_test):
        # patient_names = np.unique(np.array([i.split("/")[-3] for i in dirs_train_whole[0]]))
        # trainval_test = train_test_split(patient_names, test_size=split_rate_test, random_state=self.config.seed)
        # train_val = train_test_split(trainval_test[0], test_size=100/len(trainval_test[0]), random_state=self.config.seed)
        #
        # train_idx, val_idx, test_idx = [],[], []
        # # train_idx, val_idx, test_idx = np.arange(0,int(len(dirs_train_whole[0])*0.7)), np.arange(int(len(dirs_train_whole[0])*0.7),int(len(dirs_train_whole[0])*0.85)), np.arange(int(len(dirs_train_whole[0])*0.85),int(len(dirs_train_whole[0])))
        # for index, file_name in enumerate(dirs_train_whole[0]):
        #     if file_name.split("/")[-3] in trainval_test[1]:
        #         test_idx.append(index)
        #     elif file_name.split("/")[-3] in train_val[0]:
        #         train_idx.append(index)
        #     else:
        #         val_idx.append(index)
        # print("Train {} validation {} test {}".format(len(train_idx),len(val_idx),len(test_idx)))
        # split_dict = {"train":train_idx,"test":test_idx,"valid":val_idx}
        # with open('/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/shhs1_full_random_split.pickle', 'wb') as f:
        #     pickle.dump(split_dict, f)

        with open('/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/shhs1_random_split.pickle', 'rb') as f:
            split_dict = pickle.load(f)

        return np.array(dirs_train_whole[:,split_dict["train"]]), np.array(dirs_train_whole[:,split_dict["valid"]]), np.array(dirs_train_whole[:,split_dict["test"]])

    def _split_patients_folds(self, dirs_train_whole, fold, fold_size):
        patient_names = np.unique(np.array([i.split("/")[-3] for i in dirs_train_whole[0]]))
        np.random.seed(self.config.seed)
        np.random.shuffle(patient_names)
        chosen_names = patient_names[fold_size* fold: fold_size * (1+ fold) ]
        train_idx, val_idx = [],[]
        for index, file_name in enumerate(dirs_train_whole[0]):
            if file_name.split("/")[-3] not in chosen_names:
                train_idx.append(index)
            else:
                val_idx.append(index)
        return np.array(dirs_train_whole[:,train_idx]), np.array(dirs_train_whole[:,val_idx])

    def _split_huy_folds(self, dirs_train_whole, fold, fold_size):

        cross_val_run = pickle.load(open(self.config.folds_file, "rb"))
        fold_split = cross_val_run[fold]
        train_idx, val_idx, test_idx = [],[],[]
        for index, file_name in enumerate(dirs_train_whole[0]):
            if int(file_name.split("/")[-3][-2:]) in fold_split["train"]:
                train_idx.append(index)
            elif int(file_name.split("/")[-3][-2:]) in fold_split["eval"]:
                val_idx.append(index)
            elif int(file_name.split("/")[-3][-2:]) in fold_split["test"]:
                test_idx.append(index)
            else:
                raise Warning("Splitting is not going well, some patient has no house. \n {}".format(file_name))
        return np.array(dirs_train_whole[:,train_idx]), np.array(dirs_train_whole[:,val_idx]), np.array(dirs_train_whole[:,test_idx])

    def _split_patients_num(self, dirs_train_whole, patients_num):
        train_idx, val_idx = [],[]
        if type(patients_num) is list:
            patients_list = ["patient_{}".format(f'{p:02}') for p in patients_num]
        else:
            patients_list = ["patient_{}".format(f'{patients_num:02}')]
        for index, file_name in enumerate(dirs_train_whole[0]):
            if file_name.split("/")[-3] in patients_list:
                val_idx.append(index)
            else:
                train_idx.append(index)
        return np.array(dirs_train_whole[:,train_idx]), np.array(dirs_train_whole[:,val_idx])

    def _random_folds(self, dirs_train_whole, chosen_split, rand_splits, seed):
        skf = StratifiedKFold(n_splits=rand_splits, shuffle=True,random_state=seed)
        skf.get_n_splits(range(len(dirs_train_whole[0])), dirs_train_whole[1])
        for ind,(train_idx, val_idx) in enumerate(skf.split(range(len(dirs_train_whole[0])), dirs_train_whole[1])):
            if ind == chosen_split:
                break
        return np.array(dirs_train_whole[:, train_idx]), np.array(dirs_train_whole[:, val_idx])

    def _split_random(self, dirs_train_whole, split_rate):
        split_indices = train_test_split(range(len(dirs_train_whole[0])), test_size=split_rate, random_state=self.config.seed, stratify=dirs_train_whole[1])
        return np.array(dirs_train_whole[:,split_indices[0]]), np.array(dirs_train_whole[:,split_indices[1]])