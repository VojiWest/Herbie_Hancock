from Preprocess import preprocess
from Preprocess import augmentation
from Model import keras_model, pytorch_model
from Plots import plot
from Utils import utils

import numpy as np
import sys
import torch



def main():
    data = preprocess.load_data(path = "F.txt") # Load the data
    data = data[3] # we choose only voice 1 

    """ Preprocess data """

    train, val, test = preprocess.temporal_data_split(data)
    train_X, train_Y = preprocess.create_input_windows(train, seq_length = 50)

    
    voice_ranges = [(50, 80), (40, 70), (35, 65), (25, 55)]
    voice_range_voice1 = (50, 80)
    voice_range_voice2 = (40, 70)
    voice_range_voice3 = (35, 65)
    voice_range_voice4 = (25, 55)
    y_one_hot = preprocess.one_voice_convert_onehot(train_Y, voice_range_voice4)


    encoded_voice = augmentation.augmented_encoding(train)
    print(encoded_voice)

    # circle_of_fifths_representation = augmentation.convert_voices_circle_of_fifths(train)
    # chromatic_cirlce_representation = augmentation.convert_voices_chromatic_circle(train)

    # augmentation.plot_circle(circle_of_fifths_representation)
    # augmentation.plot_circle(chromatic_cirlce_representation)
 
if __name__ == "__main__":
    main()
