import pandas as pd
import numpy as np
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import torch

def add_preds_to_data(data, predictions):
    # Reshape predictions to remove batch dimension
    predictions_reshaped = predictions

    # Create a DataFrame with the same column names as `train`
    print("Data", data.shape)
    new_predictions = pd.DataFrame(predictions_reshaped, columns=data.columns)
    print("New Predictions", new_predictions.shape)
    new_data = pd.concat([data, new_predictions], ignore_index=True)
    print("New Data", new_data.shape)

    return new_data, new_predictions

def get_time_signature():
    # get unique label for file using current time
    now = time.localtime()
    day = now.tm_yday
    hour = now.tm_hour
    minute = now.tm_min
    unique_label = f"{day}_{hour}_{minute}"

    return unique_label


def save_file(data, name, path):
    # get unique label for file using current time
    time_signature = get_time_signature()

    np.save(path+name+time_signature+".npy", data)

def save_model(model, name, path):
    # get unique label for file using current time
    time_signature = get_time_signature()

    torch.save(model, path+name+time_signature+".pt")

def weights_to_tensor(class_weights_list):
    class_weights_per_voice = []
    for cw in class_weights_list:
        # Convert each dictionary of class weights to a tensor
        class_weights = [cw.get(i, 1.0) for i in range(max(cw.keys()) + 1)]
        class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
        class_weights_per_voice.append(class_weights_tensor)

    return class_weights_per_voice

def normalize_data(data): # Function not qutie working, need to fix
    # Normalize the data
    if isinstance(data, list):
        data = np.array(data)  # Convert list to NumPy array
    elif isinstance(data, np.ndarray):
        data = torch.tensor(data)  # Convert NumPy array to PyTorch tensor
    # Reshape to (batch_size * seq_length, num_features)
    print("Data Shape: ", data.shape)
    data_flat = data.view(-1, data.shape[-1])  # Using .shape for clarity

    # Calculate min and max per feature
    data_min = data_flat.min(dim=0, keepdim=True).values
    data_max = data_flat.max(dim=0, keepdim=True).values

    # Apply Min-Max Scaling
    data_scaled_flat = (data_flat - data_min) / (data_max - data_min + 1e-8)  # Add small epsilon to avoid division by zero

    # Reshape back to original shape
    data_scaled = data_scaled_flat.view(data.size())

    return data_scaled
