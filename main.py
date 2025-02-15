from Audio import audio_midi
from Model_LR import lr_model, trainer, evaluator, predictor
from Plotting import plot
from Utils import utils
from Dataset import dataset
from Loss import custom_CE

import numpy as np
import torch

def main():
    voice_num = 0
    
    # Set hyperparameters to those chosen during hyperparameter tuning
    window_size = 16
    k = 3
    learning_rate = 0.01

    ds_voice = dataset.CustomDataset(window_size=window_size, voice_num=voice_num, val_ratio=0) # set val_ratio to 0 to use all data for training
    output_to_input_convert = ds_voice.get_output_to_input_matching()
    non_zero_min_note, max_note = ds_voice.get_non_zero_min_and_max()
    max_duration = ds_voice.get_max_duration()

    """ Preprocess data """
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = ds_voice.get_train_val_test() 

    """ Train model """
    input_size = X_train_tensor[0].numel()
    output_size = ds_voice.get_unique_target_values() # Number of unique notes

    model = lr_model.ActualLogisticRegressionModel(input_size, output_size)
    criterion = custom_CE.CustomCrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    flat_X_train_tensor = torch.flatten(X_train_tensor, start_dim=1) # Flatten tensor
    flat_X_test_tensor = torch.flatten(X_test_tensor, start_dim=1) # Flatten tensor

    model = trainer.train_model(flat_X_train_tensor, y_train_tensor, flat_X_test_tensor, y_test_tensor, model, optimizer, criterion, plot_losses=True)

    """ Run model to Predict Bach"""
    max_pred, all_preds = predictor.predict_bach(flat_X_train_tensor[-1], model, output_to_input_convert, non_zero_min_note, max_note, max_duration, timesteps=383, k=k)

    """ Evaluate model """
    test_accuracy = utils.get_accuracy(max_pred, ds_voice.get_test())
    test_mae = utils.get_mae(max_pred, ds_voice.get_test())

    print("Test Accuracy: ", test_accuracy, "Test MAE: ", test_mae)

    plot.plot_histogram(max_pred, title="Predicted Notes Histogram", xlabel="Note", ylabel="Frequency", one_voice=True)
    plot.plot_histogram(ds_voice.get_test(), title="Test Data Note Frequency", xlabel="Note", ylabel="Frequency", one_voice=True)

    """ Postprocess data """

    all_data = np.array(ds_voice.get_train()) # For one voice

    new_data, new_predictions = utils.add_preds_to_data(all_data, max_pred)
    new_data = new_data.astype(int)


    plot.plot_certainty(all_preds, title = "Certainty of Predictions", xlabel = "Time", ylabel = "Note") # Plot certainty of each note over timesteps
    plot.plot_data(all_data, title = "Original Data", xlabel = "Time", ylabel = "Note") # plot original notes
    plot.plot_data(new_data, title = "Original + Predicted Data", xlabel = "Time", ylabel = "Note") # plot the original + predicted notes
    plot.plot_data(new_predictions, title = "Predicted Data", xlabel = "Time", ylabel = "Note") # plot predicted notes

    audio_midi.data_to_audio(new_data, "Final LR original + predictions", one_voice=True)
    audio_midi.data_to_audio(max_pred, "Final LR just predictions", one_voice=True)

    

if __name__ == "__main__":
    main()