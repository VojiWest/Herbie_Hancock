# prompt: make pytorch dataset class

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path='F.txt', window_size=10, voice_num=0, test_split=0.2):
        self.data = pd.read_csv(data_path, sep='\t', header=None)
        self.data = self.data[voice_num]
        self.train, self.test = self.split_temporal_data(self.data, test_split)
        self.X_train, self.y_train = self.create_sliding_window_dataset(self.train, window_size)
        self.X_test, self.y_test = self.create_sliding_window_dataset(self.test, window_size)
        self.y_train = self.encode_labels(self.y_train)
        self.y_test = self.encode_labels(self.y_test)

    def split_temporal_data(self, data, test_size):
      print("Splitting train test")
      train_size = int(len(data) * (1-test_size))  # 80% for training, 20% for testing
      train_data, test_data = data[:train_size], data[train_size:]
      # drop indices
      train_data = train_data.reset_index(drop=True)
      test_data = test_data.reset_index(drop=True)
      return train_data, test_data

    def create_sliding_window_dataset(self, data, window_size):
      print("Creating sliding window dataset")
      X, y = [], []
      for i in range(len(data) - window_size):
          X.append(data[i:i + window_size])
          y.append(data[i + window_size])
      return np.array(X), np.array(y)

    def encode_labels(self, labels):
      # one-hot encodes labels
      # restricts the possible note values to the notes that have already occured (no notes outside this range)
      min_val = 0
      max_val = max(self.data)
      unique_labels = np.arange(min_val, max_val + 1)
      num_classes=len(unique_labels)
      encoded_labels = np.zeros((len(labels), num_classes))
      for i, label in enumerate(labels):
          encoded_labels[i, label] = 1
      return encoded_labels

    def get_train_test(self):
      return self.X_train, self.y_train, self.X_test, self.y_test

    def __len__(self):
        return len(self.train), len(self.test)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

    def get_y_train(self):
      return self.y_train

    def get_unique_target_values(self):
      min_val = 0
      max_val = max(self.data)
      unique_labels = np.arange(min_val, max_val + 1)
      num_classes=len(unique_labels)
      return num_classes