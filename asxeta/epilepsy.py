import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Seizure_Dataset(Dataset):
    def __init__(self, filename):
        self.hdf5_filename = filename
        with h5py.File(filename, "r") as f:
            self.filenames = list(f['filenames'])
    def __len__(self):
        return 5

    def __getitem__(self, index):
        with h5py.File(self.hdf5_filename, "r") as f:
            labels = list(f["labels"][index])
            signals = list(f["signals"][index])
        return labels, signals

a = Seizure_Dataset(filename = "/esat/stadiustempdatasets/SeizeIT1/Data/SZ1_full_processed_data.hdf5")
label, signal = a.__getitem__(0)

print(np.array(label).shape)
print(np.array(signal).shape)