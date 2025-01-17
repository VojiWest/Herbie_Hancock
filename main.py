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
    ds_voice_1 = dataset.CustomDataset(window_size=window_size) # Added augmentation application into the dataset creation
    output_to_input_convert = ds_voice_1.get_output_to_input_matching()

    """ Preprocess data """
    X_train, y_train, X_test, y_test = ds_voice_1.get_train_test() 

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # class_weights = preprocess.get_class_weights(y_one_hot)


    """ Train and run model """
    # Hyperparameters
    input_size = X_train_tensor[0].numel()
    output_size = ds_voice_1.get_unique_target_values() # Number of unique notes
    print("Num Classes: ", output_size)
    learning_rate = 0.001

    model = lr_model.ActualLogisticRegressionModel(input_size, output_size)
    criterion = custom_CE.CustomCrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("X Train Shape:", X_train_tensor.shape)
    flat_X_train_tensor = torch.flatten(X_train_tensor, start_dim=1) # Flatten tensor
    print("y Train Shape:", y_train_tensor.shape)

    model = trainer.train_model(flat_X_train_tensor, y_train_tensor, model, optimizer, criterion)
    
    preds = predictor.predict_bach(flat_X_train_tensor[-1], model, output_to_input_convert)

    """ Postprocess data """

    new_data, new_predictions = utils.add_preds_to_data(ds_voice_1.get_train(), preds)

    # plot.plot_certainty(all_preds, title = "Certainty of Predictions", xlabel = "Time", ylabel = "Note")
    plot.plot_data(new_data, title = "Original + Predicted Data", xlabel = "Time", ylabel = "Note")
    plot.plot_data(new_predictions, title = "Predicted Data", xlabel = "Time", ylabel = "Note")

    audio_midi.data_to_audio(new_data, "LR full pred")


    

if __name__ == "__main__":
    main()