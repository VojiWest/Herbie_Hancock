from Dataset.augmentation import augmented_encoding
from Dataset.dataset import CustomDataset

import torch
import numpy as np

def sample_from_topk(output, k):
    # Get the top-k predictions
    topk_probs, topk_indices = torch.topk(output, k)
    # create a categorical distribution based on the top-k probabilities using softmax with temperature
    topk_probs = torch.softmax(topk_probs/0.2, dim=1)
    # print("output: ", output)
    # print("topk_probs: ", topk_probs)
    

    # select an index based on top-n probabilities
    sampled_index = torch.multinomial(topk_probs, num_samples=1).item()
    chosen_index = topk_indices[0, sampled_index]
    # print("chosen_index: ", chosen_index, "  sampled_index: ", sampled_index)

    return chosen_index

def plot_raw_output(output, title):
    import matplotlib.pyplot as plt
    plt.plot(output)
    plt.title(title)
    plt.show()


# k is how many top predictions we randomly sample from. This can be tuned
def predict_bach(last_timestep, model, output_to_input_converter, note_min, note_max, max_duration, timesteps = 400, k = 3): 
    max_prediction = []
    all_predictions = []
    with torch.no_grad():
        for step in range(timesteps):
            last_timestep_twod = last_timestep.unsqueeze(0)  # Shape becomes [1, 22]
            output = model(last_timestep_twod)
            all_predictions.append(output)
            output_max = torch.argmax(output) # for just getting the max pred

            ### Sampling ###
            sampled_index = sample_from_topk(output, k)

            # Output is now from 0-22, but to feed back into data augmentation we need to 
            # convert it back into the MIDI-like format form 0-127
            raw_input_domain_output = [output_to_input_converter[sampled_index]] # set index to chosen_index or output_max in the inner most brackets
            aug_output = augmented_encoding(raw_input_domain_output, note_min, note_max, max_duration)
            aug_X_tensor = torch.tensor(aug_output, dtype=torch.float32)
            aug_X_tensor = torch.flatten(aug_X_tensor)

            last_timestep = torch.cat((last_timestep[6:], aug_X_tensor), 0) # Remove the first 6 elements and add the new prediction (6 since we have 6 features)
            max_prediction.append(raw_input_domain_output[0])

    return np.array(max_prediction), torch.stack(all_predictions).numpy()


