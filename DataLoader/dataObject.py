import os
import pickle

import mne
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


class DataObject:
    TRAINING_DATA_PATH = r'data/training/'
    TEST_DATA_PATH = r'data/testing_flat/'
    CACHE_PATH = r'cache/'
    STATES = ['EO', 'EC', 'ALL']

    def __init__(self,
                 data_cache_id: str,
                 subject_count: int,
                 state: str,
                 crop_start: int,
                 crop_end: int,
                 load_cache: bool = False):

        self.cache_fname = data_cache_id + "_" + state + "_" + str(subject_count) + "_" + str(crop_start) + "-" + str(crop_end)

        self.state = state
        self.subject_count = subject_count
        self.crop_start = crop_start
        self.crop_end = crop_end
        self.cache_exists = False

        self.x_train = torch.tensor([])
        self.x_target = torch.tensor([])

        if load_cache:
            self.load_cache()

    def load_eeg_data(self):
        """
        Loads data from files
        """
        if self.cache_exists:
            return self.x_train, self.x_target
        condition = self.state
        print(f"INFO: Loading {self.subject_count} subjects with {self.state} from file")
        print(f"INFO: Date is cropped from {self.crop_start} to {self.crop_end}")

        X_train = []
        x_target = []
        df = pd.read_csv(self.TRAINING_DATA_PATH + 'train_subjects.csv')
        for s in range(1, self.subject_count + 1):
            fname = f"subj{s:04}_{condition}_raw.fif.gz"
            raw = mne.io.read_raw(self.TRAINING_DATA_PATH + fname, preload=True)
            data = np.array(raw.get_data(tmin=self.crop_start, tmax=self.crop_end))
            if len(data[0]) == int((self.crop_end - self.crop_start) * 500):
                X_train.append(torch.tensor(data,
                                            dtype=torch.float32))
                x_target.append(torch.tensor(int(df[df["id"] == s].iloc[0]["age"])))
            else:
                print("WARNING: Subject has wrong frequency", int((self.crop_end - self.crop_start) * 500), len(data[0]))

        self.x_target = torch.tensor(np.array(x_target),
                                     dtype=torch.float32)
        self.x_train = torch.stack(X_train)

        self.save_cache()

        return self.x_train, self.x_target

    def load_eeg_test_data(self, test_subj: int):
        condition = self.state
        # number of test subjects to load
        test_subj = test_subj

        test_raws = []
        for s in range(1201, 1201 + test_subj):
            fname = f"subj{s:04}_{condition}_raw.fif.gz"
            raw = mne.io.read_raw(self.TEST_DATA_PATH + fname, preload=True)
            test_raws.append(raw)

        # Get ndarray from MNE raw files to generate train input
        X_test = []
        # use only a 10s window, from 5s to 15s
        crop_start, crop_end = 5, 15
        for r in test_raws:
            X_test.append(r.copy().crop(tmin=crop_start, tmax=crop_end).get_data())

    def load_cache(self):
        if os.path.exists(self.CACHE_PATH + self.cache_fname + '.pickle'):
            self.cache_exists = True
            with open(self.CACHE_PATH + self.cache_fname + '.pickle', "rb") as f:
                cache_data = pickle.load(f)

            self.x_train = cache_data["train_data"]
            self.x_target = cache_data["targets"]
            print("INFO: Loaded from cache.")
        else:
            print("INFO: Cache dose not exists.")

    def save_cache(self):
        cache_data = {
            "train_data": self.x_train,
            "targets": self.x_target
        }
        with open(self.CACHE_PATH + self.cache_fname + '.pickle', 'wb') as f:
            pickle.dump(cache_data, f)
