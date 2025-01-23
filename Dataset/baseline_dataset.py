import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path='F.txt', window_size=10, voice_num=0, val_ratio=0.1, test_ratio=0.1):
        self.data_all = pd.read_csv(data_path, sep='\t', header=None)
        self.data = self.data_all[voice_num]
        self.train, self.val, self.test = self.split_temporal_data(self.data, val_ratio=val_ratio, test_ratio=test_ratio)

        self.max_duration = self.get_max_duration()
        self.non_zero_min, self.max = self.get_non_zero_min_and_max()
        #self.train_aug = augmented_encoding(self.train, self.non_zero_min, self.max, self.max_duration)
        #self.val_aug = augmented_encoding(self.val, self.non_zero_min, self.max, self.max_duration)
        #self.test_aug = augmented_encoding(self.test, self.non_zero_min, self.max, self.max_duration)

        self.X_train, self.y_train = self.create_sliding_window_dataset(self.train, self.train, window_size)
        self.X_val, self.y_val = self.create_sliding_window_dataset(self.val, self.val, window_size)
        self.X_test, self.y_test = self.create_sliding_window_dataset(self.test, self.test, window_size)

        self.y_train = self.encode_labels(self.y_train)
        self.y_val = self.encode_labels(self.y_val)
        self.y_test = self.encode_labels(self.y_test)

    def split_temporal_data(self, data, val_ratio, test_ratio):
      train_size = int(len(data) * (1-(val_ratio + test_ratio)))  # Get training size
      val_size = int(len(data) * val_ratio)
      test_size = int(len(data) * test_ratio)
      print("Train Size:", train_size, " - Val Size:", val_size, " - Test Size: ", test_size)
      train_data, val_data, test_data = data[:train_size], data[train_size:(train_size + val_size)], data[test_size:]
      # drop indices
      train_data = train_data.reset_index(drop=True)
      val_data = val_data.reset_index(drop=True)
      test_data = test_data.reset_index(drop=True)
      return train_data, val_data, test_data

    def create_sliding_window_dataset(self, data_raw, data_aug, window_size):
      # print("Creating sliding window dataset")
      X, y = [], []
      for i in range(len(data_raw) - window_size):
          X.append(data_aug[i:i + window_size])
          y.append(data_raw[i + window_size])
      return np.array(X), np.array(y)

    def encode_labels(self, labels):
      # one-hot encodes labels
      # restricts the possible note values to the notes that have already occured (no notes outside this range)
      uniques = np.unique(self.train)
      # print("Uniques: ", uniques)
      num_unique = len(uniques)
      encoded_labels = np.zeros((len(labels), num_unique))
      for i, label in enumerate(labels):
              encoded_labels[i, np.where(uniques == label)] = 1  # Now it should be scaled so output classes are starting at 0
      return encoded_labels
    
    def get_max_duration(self):
        max_duration = 1
        temp_max_duration = 0
        prev_note = self.train[0]
        for index in range(1, len(self.train)):
            note = self.train[index]
            if note == prev_note:
                temp_max_duration += 1
            else:
                temp_max_duration = 0

            if temp_max_duration > max_duration:
              max_duration = temp_max_duration

        return max_duration
       
    def get_train_val_test(self):
      torch_X_train = torch.tensor(self.X_train, dtype=torch.float32)
      torch_y_train = torch.tensor(self.y_train, dtype=torch.float32)
      torch_X_val = torch.tensor(self.X_val, dtype=torch.float32)
      torch_y_val = torch.tensor(self.y_val, dtype=torch.float32)
      torch_X_test = torch.tensor(self.X_test, dtype=torch.float32)
      torch_y_test = torch.tensor(self.y_test, dtype=torch.float32)
      return torch_X_train, torch_y_train, torch_X_val, torch_y_val, torch_X_test, torch_y_test

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
    
    def get_val(self):
       return self.val
    
    def get_all_voices_data(self):
       return self.data_all
    
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