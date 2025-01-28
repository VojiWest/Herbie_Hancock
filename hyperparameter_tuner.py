from Audio import audio_midi
from Model_LR import lr_model, trainer, evaluator, predictor
from Plotting import plot
from Utils import utils
from Dataset import dataset
from Loss import custom_CE

import numpy as np
import torch

def hyperparameter_tuner():
    voice_predictions = []
    voice_num = 0

    # Do hyperparameter tuning
    parameter_search_space = { "k" : [1, 3, 6, 11], "window_size" : [16, 32, 64, 128], "learning_rate" : [0.001, 0.01, 0.1]}
    combinations = utils.get_parameter_combinations(parameter_search_space)
    
    hyperparameter_results = []
    for num, combo in enumerate(combinations):
        print(f"Hyperparameter Combination {num+1}/{len(combinations)}")

        # Set hyperparameters to evaluate
        window_size = combo["window_size"]
        k = combo["k"]
        learning_rate = combo["learning_rate"]

        # for voice_num in range(number_of_voices):
        ds_voice = dataset.CustomDataset(window_size=window_size, voice_num=voice_num) # Added augmentation application into the dataset creation
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
        flat_X_val_tensor = torch.flatten(X_val_tensor, start_dim=1) # Flatten tensor

        model = trainer.train_model(flat_X_train_tensor, y_train_tensor, flat_X_val_tensor, y_val_tensor, model, optimizer, criterion)

        """ Evaluate model """
        predictions, all_class_predictions = predictor.predict_bach(flat_X_train_tensor[-1], model, output_to_input_convert, non_zero_min_note, max_note, max_duration, timesteps=382, k=k)

        val_accuracy = utils.get_accuracy(predictions, ds_voice.get_val())
        val_mae = utils.get_mae(predictions, ds_voice.get_val())

        """ Postprocess Predictions """
        all_data = np.array(ds_voice.get_train()) # Get the original training data
        train_plus_prediction, _ = utils.add_preds_to_data(all_data, predictions)

        title = f" nND Window Size = {window_size} - Learning Rate = {learning_rate} - K = {k}"
        audio_midi.data_to_audio(train_plus_prediction, "Full --- " + title, one_voice=True, folder="Grid Search Outputs II/", hyperparameter_tuning=True)
        audio_midi.data_to_audio(predictions, "Predictions --- " + title, one_voice=True, folder="Grid Search Outputs II/", hyperparameter_tuning=True)

        hyperparameter_results.append({"Model": title, "Val Acc": val_accuracy, "Val MAE": val_mae})


    print("Hyperparameter Results: ", hyperparameter_results)
    

if __name__ == "__main__":
    hyperparameter_tuner()