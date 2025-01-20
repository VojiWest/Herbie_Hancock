# prompt: make pytorch dataset class
from .augmentation import augmented_encoding

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path='F.txt', window_size=10, voice_num=0, test_split=0.2):
        self.data = pd.read_csv(data_path, sep='\t', header=None)
        self.data = self.data[voice_num]
        self.train, self.test = self.split_temporal_data(self.data, test_split)
        self.non_zero_min, self.max = self.get_non_zero_min_and_max()
        self.train_aug = augmented_encoding(self.train, self.non_zero_min, self.max)
        self.test_aug = augmented_encoding(self.test, self.non_zero_min, self.max)
        self.X_train, self.y_train = self.create_sliding_window_dataset(self.train, self.train_aug, window_size)
        self.X_test, self.y_test = self.create_sliding_window_dataset(self.test, self.test_aug, window_size)
        self.y_train = self.encode_labels(self.y_train)
        self.y_test = self.encode_labels(self.y_test)

    def split_temporal_data(self, data, test_size):
      print("Splitting train test")
      train_size = int(len(data) * (1-test_size))  # 80% for training, 20% for testing
      train_data, test_data = data[:train_size], data[train_size:]
      # drop indices
      train_data = train_data.reset_index(drop=True)
      test_data = test_data.reset_index(drop=True)
      print("train data size: ", train_data.shape)
      return train_data, test_data

    def create_sliding_window_dataset(self, data_raw, data_aug, window_size):
      print("Creating sliding window dataset")
      X, y = [], []
      for i in range(len(data_raw) - window_size):
          X.append(data_aug[i:i + window_size])
          y.append(data_raw[i + window_size])
      return np.array(X), np.array(y)

    def encode_labels(self, labels):
      # one-hot encodes labels
      # restricts the possible note values to the notes that have already occured (no notes outside this range)
      uniques = np.unique(self.train)
      print("Uniques: ", uniques)
      num_unique = len(uniques)
      encoded_labels = np.zeros((len(labels), num_unique))
      for i, label in enumerate(labels):
              encoded_labels[i, np.where(uniques == label)] = 1  # Now it should be scaled so output classes are starting at 0
      return encoded_labels

    def get_train_test(self):
      return self.X_train, self.y_train, self.X_test, self.y_test

    def __len__(self):
        return len(self.train), len(self.test)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

    def get_y_train(self):
      return self.y_train
    
    def get_X_train(self):
       return self.X_train
    
    def get_train(self):
       return self.train
    
    def get_output_to_input_matching(self):
      uniques = np.unique(self.train)
      return uniques
    
    def get_non_zero_min_and_max(self):
      non_zero_vals = [i for i in self.train if i != 0]
      min_non_zero_note = min(non_zero_vals)
      max_note = max(non_zero_vals)

      return min_non_zero_note, max_note

    def get_unique_target_values(self):
      num_unique = len(np.unique(self.train))
      return num_unique