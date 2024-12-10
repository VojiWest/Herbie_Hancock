import pandas as pd
import numpy as np

def load_data(path):
    path = "F.txt"

    # Load the data
    data = pd.read_csv(path, sep="\t", header=None)

    return data

def temporal_data_split(data, train = 0.7, val = 0.15, test = 0.15):
    train_split = data[: round(len(data) * train)]
    val_split = data[round(len(data) * train) : round(len(data) * (train + val))]
    test_split = data[round(len(data) * (train + val)) :]

    return train_split, val_split, test_split

def create_input_windows(data, seq_length):
    X = []
    y = []

    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i + seq_length].values)  # Use `.iloc` for slicing
        y.append(data.iloc[i + seq_length])          # Use `.iloc` for a single value

    X = np.array(X)
    y = np.array(y)

    return X, y