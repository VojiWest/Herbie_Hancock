from Audio import audio_midi
from Model_LR import lr_model, trainer, evaluator, predictor
from Plotting import plot
from Utils import utils
from Dataset import baseline_dataset
from Loss import custom_CE
from CVGridSearch import cvgridsearch

import numpy as np
import torch

def main():
    voice_predictions = []
    voice_num = 0

    # Do hyperparameter tuning
    #parameter_search_space = { "k" : [3, 6, 11], "window_size" : [16, 32, 64, 128], "learning_rate" : [0.001, 0.01, 0.1]}
    parameter_search_space = { "k" : [1], "window_size" : [128], "learning_rate" : [0.1]}
    combinations = utils.get_parameter_combinations(parameter_search_space)
    
    for combo in combinations:
        # Set hyperparameters to evaluate
        window_size = combo["window_size"]
        k = combo["k"]
        learning_rate = combo["learning_rate"]

        # for voice_num in range(number_of_voices):
        ds_voice = baseline_dataset.CustomDataset(window_size=window_size, voice_num=voice_num, val_ratio=0) # Added augmentation application into the dataset creation
        output_to_input_convert = ds_voice.get_output_to_input_matching()
        non_zero_min_note, max_note = ds_voice.get_non_zero_min_and_max()
        max_duration = ds_voice.get_max_duration()

        """ Preprocess data """
        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = ds_voice.get_train_val_test() 
        print("X_train_tensor", X_train_tensor.shape)
        print("y_train_tensor", y_train_tensor.shape)

        """ Train model """
        input_size = X_train_tensor[0].numel()
        output_size = ds_voice.get_unique_target_values() # Number of unique notes

        model = lr_model.ActualLogisticRegressionModel(input_size, output_size)
        criterion = custom_CE.CustomCrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        flat_X_train_tensor = torch.flatten(X_train_tensor, start_dim=1) # Flatten tensor
        flat_X_test_tensor = torch.flatten(X_test_tensor, start_dim=1) # Flatten tensor
        print("flat_X_train_tensor", flat_X_train_tensor.shape)
        print("flat x val shape",flat_X_test_tensor.shape)

        model = trainer.train_model(flat_X_train_tensor, y_train_tensor, flat_X_test_tensor, y_test_tensor, model, optimizer, criterion)

        """ Evaluate model """
        predictions, _ = predictor.predict_bach(flat_X_train_tensor[-1], model, output_to_input_convert, non_zero_min_note, max_note, max_duration, timesteps=383, k=k)

        test_accuracy = utils.get_accuracy(predictions, ds_voice.get_test())
        test_mae = utils.get_mae(predictions, ds_voice.get_test())

        """ Postprocess Predictions """
        all_data = np.array(ds_voice.get_train()) # Get the original training data
        train_plus_prediction, _ = utils.add_preds_to_data(all_data, predictions)

        title = f" Window Size = {window_size} - Learning Rate = {learning_rate} - K = {k}"
        audio_midi.data_to_audio(train_plus_prediction, "Full --- " + title, one_voice=True, folder="Grid Search Outputs/")
        audio_midi.data_to_audio(predictions, "Predictions --- " + title, one_voice=True, folder="Grid Search Outputs/")

        print("Model:  ", title, "  --- Test Acc: ", test_accuracy, "  Test MAE: ", test_mae)

    """ Run model to Predict Bach"""
    max_pred, all_preds = predictor.predict_bach(flat_X_train_tensor[-1], model, output_to_input_convert, non_zero_min_note, max_note, max_duration)

    voice_predictions.append(max_pred)
    print("Len voice", voice_num, ":", len(max_pred))

    """ Postprocess data """

    # max_pred = utils.combine_voices(voice_predictions) # If multiple voice use this (it may cause errors tho)

    # all_data = ds_voice.get_all_voices_data() # For multiple voices
    all_data = np.array(ds_voice.get_train()) # For one voice

    new_data, new_predictions = utils.add_preds_to_data(all_data, max_pred)
    new_data = new_data.astype(int)


    plot.plot_certainty(all_preds, title = "Certainty of Predictions", xlabel = "Time", ylabel = "Note") # Plot certainty of each note over timesteps
    plot.plot_data(all_data, title = "Original Data", xlabel = "Time", ylabel = "Note") # plot original notes
    plot.plot_data(new_data, title = "Original + Predicted Data", xlabel = "Time", ylabel = "Note") # plot the original + predicted notes
    plot.plot_data(new_predictions, title = "Predicted Data", xlabel = "Time", ylabel = "Note") # plot predicted notes

    audio_midi.data_to_audio(new_data, "LogReg Voice Zero original + predictions", one_voice=True)
    audio_midi.data_to_audio(max_pred, "LogReg Voice Zero just predictions", one_voice=True)


    

if __name__ == "__main__":
    main()