from Dataset.augmentation import augmented_encoding
from Dataset.dataset import CustomDataset

import torch
import numpy as np

def predict_bach(last_timestep, model, output_to_input_converter, note_min, note_max, timesteps = 400):
    max_prediction = []
    all_predictions = []
    with torch.no_grad():
        for step in range(timesteps):
            last_timestep_twod = last_timestep.unsqueeze(0)  # Shape becomes [1, 22]
            output = model(last_timestep_twod)
            all_predictions.append(output)
            output_max = torch.argmax(output)

            # Output is now from 0-22, but to feed back into data augmentation we need to 
            # convert it back into the MIDI-like format form 0-127
            raw_input_domain_output = [output_to_input_converter[output_max]]
            aug_output = augmented_encoding(raw_input_domain_output, note_min, note_max)
            aug_X_tensor = torch.tensor(aug_output, dtype=torch.float32)
            aug_X_tensor = torch.flatten(aug_X_tensor)

            last_timestep = torch.cat((last_timestep[6:], aug_X_tensor), 0)
            max_prediction.append(raw_input_domain_output[0])

    return np.array(max_prediction), np.array(all_predictions)


