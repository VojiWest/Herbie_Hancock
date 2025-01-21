from Audio import audio_midi
from Preprocess import preprocess
from Model import keras_model, pytorch_model, logistic_reg
from Model_LR import ff_model, lr_model, trainer, evaluator, predictor
from Plots import plot
from Utils import utils
from Dataset import dataset
from Loss import custom_CE



import numpy as np
import sys
import torch

def main():
    window_size = 64
    number_of_voices = 0

    voice_predictions = []
    voice_num = 0

    # for voice_num in range(number_of_voices):
    ds_voice = dataset.CustomDataset(window_size=window_size, voice_num=voice_num) # Added augmentation application into the dataset creation
    output_to_input_convert = ds_voice.get_output_to_input_matching()
    non_zero_min_note, max_note = ds_voice.get_non_zero_min_and_max()

    """ Preprocess data """
    X_train, y_train, X_val, y_val, X_test, y_test = ds_voice.get_train_val_test() 

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    """ Train model """
    # Hyperparameters
    input_size = X_train_tensor[0].numel()
    output_size = ds_voice.get_unique_target_values() # Number of unique notes
    print("Num Classes: ", output_size)
    learning_rate = 0.001
    hidden_size = 64

    model = lr_model.ActualLogisticRegressionModel(input_size, output_size)
    # model = ff_model.LogisticRegressionModel(input_size, hidden_size, output_size)
    criterion = custom_CE.CustomCrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    flat_X_train_tensor = torch.flatten(X_train_tensor, start_dim=1) # Flatten tensor
    flat_X_val_tensor = torch.flatten(X_val_tensor, start_dim=1) # Flatten tensor

    model = trainer.train_model(flat_X_train_tensor, y_train_tensor, flat_X_val_tensor, y_val_tensor, model, optimizer, criterion)

    """ Run model to Predict Bach"""
    max_pred, all_preds = predictor.predict_bach(flat_X_train_tensor[-1], model, output_to_input_convert, non_zero_min_note, max_note)
    print(max_pred)

    voice_predictions.append(max_pred)
    print("Len voice", voice_num, ":", len(max_pred))

    """ Postprocess data """

    # max_pred = utils.combine_voices(voice_predictions) # If multiple voice use this (it may cause errors tho)

    # all_data = ds_voice.get_all_voices_data() # For multiple voices
    all_data = np.array(ds_voice.get_train()) # For one voice

    new_data, new_predictions = utils.add_preds_to_data(all_data, max_pred)
    new_data = new_data.astype(int)

    print("new data" , new_data)

    plot.plot_certainty(all_preds, title = "Certainty of Predictions", xlabel = "Time", ylabel = "Note") # Plot certainty of each note over timesteps
    plot.plot_data(all_data, title = "Original Data", xlabel = "Time", ylabel = "Note") # plot original notes
    plot.plot_data(new_data, title = "Original + Predicted Data", xlabel = "Time", ylabel = "Note") # plot the original + predicted notes
    plot.plot_data(new_predictions, title = "Predicted Data", xlabel = "Time", ylabel = "Note") # plot predicted notes

    audio_midi.data_to_audio(new_data, "LogReg Voice Zero original + predictions", one_voice=True)
    audio_midi.data_to_audio(max_pred, "LogReg Voice Zero just predictions", one_voice=True)


    

if __name__ == "__main__":
    main()