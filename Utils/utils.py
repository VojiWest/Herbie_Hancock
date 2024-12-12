import pandas as pd
import numpy as np
import time
import tensorflow as tf

def add_preds_to_data(data, predictions):
    # Reshape predictions to remove batch dimension
    predictions_reshaped = predictions

    # Create a DataFrame with the same column names as `train`
    new_predictions = pd.DataFrame(predictions_reshaped, columns=data.columns)
    new_data = pd.concat([data, new_predictions], ignore_index=True)

    return new_data, new_predictions

def save_file(data, name, path):
    # get unique label for file using current time
    now = time.localtime()
    day = now.tm_yday
    hour = now.tm_hour
    minute = now.tm_min
    unique_label = f"{day}_{hour}_{minute}"

    np.save(path+name+unique_label+".npy", data)

def weights_to_tensor(class_weights_list):
    class_weights_per_voice = []
    for cw in class_weights_list:
        # Convert each dictionary of class weights to a tensor
        class_weights = [cw.get(i, 1.0) for i in range(max(cw.keys()) + 1)]
        class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
        class_weights_per_voice.append(class_weights_tensor)

    return class_weights_per_voice