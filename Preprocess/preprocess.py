import pandas as pd
import numpy as np
from sklearn.utils import class_weight

def load_data(path):

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

def convert_voices_onehot(data, voice_ranges):
    # For each voice, convert the data to one-hot encoding
    one_hot_data = []
    for i in range(data.shape[1]):
        one_hot_voice = []
        for j in range(len(data)):
            one_hot = np.zeros(31)
            if data[j, i] == 0:
                one_hot[0] = 1
            else:
                one_hot[data[j, i] - voice_ranges[i][0]] = 1
            one_hot_voice.append(one_hot)
        one_hot_data.append(one_hot_voice)

    y_one_hot = np.array(one_hot_data)
    y_one_hot = y_one_hot.reshape(y_one_hot.shape[1], y_one_hot.shape[0], y_one_hot.shape[2])
    
    return y_one_hot

def get_class_weights(labels, num_classes=31):
    # Calculate the class weights for each voice
    # Labels is a 3D array with shape (num_samples, num_voices, num_classes)
    all_class_weights = []
    labels = labels.reshape(labels.shape[1], labels.shape[0], labels.shape[2])
    print(labels.shape)
    label_1, label_2, label_3, label_4 = labels[0], labels[1], labels[2], labels[3] # labels are in the shape (num_samples, num_classes)
    
    for voice_idx in range(labels.shape[0]):
        # Convert one-hot encoded labels to class indices (1D array)
        output_labels = np.argmax(labels[voice_idx], axis=1)
        
        # Compute class weights for this voice
        unique_classes = np.unique(output_labels)
        
        # Compute the class weights for only the unique classes present in the data
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,  # Only pass the unique classes that are present
            y=output_labels
        )
        
        # Create a dictionary mapping class index to its weight
        # Fill in weights for all classes, even those not present in the data
        class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}
        
        # Ensure all classes have a weight, even if not present in this particular set of labels
        # For classes not present, assign a default weight (e.g., 1.0)
        full_class_weights = {cls: class_weight_dict.get(cls, 1.0) for cls in range(num_classes)}
        
        all_class_weights.append(full_class_weights)
    
    return all_class_weights

def get_first_voice(data_X, data_y):
    # Get the first voice from the data
    print("Data X Shape: ", data_X.shape, "Data Y Shape: ", data_y.shape)
    data_X_first_voice = data_X[:, :, 0]
    data_y_first_voice = data_y[:, 0, :]
    print("Data X First Voice Shape: ", data_X_first_voice.shape, "Data Y First Voice Shape: ", data_y_first_voice.shape)
    
    return data_X_first_voice, data_y_first_voice