from Audio import audio_midi
from Preprocess import preprocess
from Model import keras_model, pytorch_model
from Plots import plot
from Utils import utils

import numpy as np
import sys
import torch

def train_and_run_keras_model(mode, train_X, y_one_hot, voice_ranges, class_weights, seq_length = 16): # Not used
    if mode == "train":
        model = keras_model.create_model(train_X[0].shape, class_weights)
        model = keras_model.train_model(model, train_X, y_one_hot, epochs = 100)

        predictions, all_preds = keras_model.predict(model, train_X, voice_ranges, seq_length = seq_length, extension_length = 500)

        # save predictions to a file
        utils.save_file(predictions, "keras_predictions", "Model Outputs/")
        utils.save_file(all_preds, "keras_all_preds", "Model Outputs/")

    elif mode == "eval":
        predictions = np.load("Model Outputs/keras_predictions.npy")
        all_preds = np.load("Model Outputs/keras_all_preds.npy")

    return predictions, all_preds

def train_and_run_pytorch_model(mode, train_X, y_one_hot, voice_ranges, class_weights, seq_length = 16):
    if mode == "train":
        # Convert things to tensors since PyTorch works with tensors
        class_weights = utils.weights_to_tensor(class_weights)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
        train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
        y_one_hot = torch.tensor(y_one_hot, dtype=torch.float32).to(device)

        # Create model, optimizer and loss function. Give the model the input shape and class weights
        model, optimizer, loss_fn = pytorch_model.create_model(train_X[0].shape, class_weights)
        # Train the model
        model = pytorch_model.train_model(model, optimizer, loss_fn, train_X, y_one_hot, epochs = 500)

        # Save the model
        utils.save_model(model, "pytorch_model", "Saved Models/")

        # Run model to generate predictions to extend the Bach piece (Extend Bach piece)
        predictions, all_preds = pytorch_model.predict(model, train_X, voice_ranges, seq_length = seq_length, extension_length = 400) # Extension length is number of timesteps to extend the Bach piece by

        # save predictions to a file
        utils.save_file(predictions, "pytorch_predictions", "Model Outputs/")
        utils.save_file(all_preds, "pytorch_all_preds", "Model Outputs/")

    elif mode == "predict":
        # Load model
        model = torch.load("Saved Models/pytorch_model352_18_7.pt")

        # Run model to generate predictions to extend the Bach piece (Extend Bach piece)
        predictions, all_preds = pytorch_model.predict(model, train_X, voice_ranges, seq_length = seq_length, extension_length = 400) # Extension length is number of timesteps to extend the Bach piece by

        # save predictions to a file
        utils.save_file(predictions, "pytorch_predictions", "Model Outputs/")
        utils.save_file(all_preds, "pytorch_all_preds", "Model Outputs/")

    elif mode == "eval":
        predictions = np.load("Model Outputs/pytorch_predictions346_17_38.npy")
        all_preds = np.load("Model Outputs/pytorch_all_preds346_17_38.npy")

    return predictions, all_preds

def main(mode):
    data = preprocess.load_data(path = "F.txt") # Load the data

    ### Plotting the  
    # plot.plot_data(data, title = "Original Data", xlabel = "Time", ylabel = "Note")
    # plot.plot_histogram(data, title = "Histogram of Original Data", xlabel = "Note", ylabel = "Count")

    # data_to_audio(data, "original")

    """ Preprocess data """

    train, val, test = preprocess.temporal_data_split(data)
    train_X, train_Y = preprocess.create_input_windows(train, seq_length = 50)

    voice_ranges = [(50, 80), (40, 70), (35, 65), (25, 55)]
    y_one_hot = preprocess.convert_voices_onehot(train_Y, voice_ranges)

    # plot the one hot data
    # plot.plot_one_hot_labels(y_one_hot, title = "One Hot Labels", xlabel = "Time", ylabel = "Note")

    class_weights = preprocess.get_class_weights(y_one_hot)

    # plot.plot_class_weights(utils.weights_to_tensor(class_weights))

    """ Train and run model """

    predictions, all_preds = train_and_run_pytorch_model(mode, train_X, y_one_hot, voice_ranges, class_weights, seq_length = 50)

    """ Postprocess data """

    new_data, new_predictions = utils.add_preds_to_data(train, predictions)

    plot.plot_certainty(all_preds, title = "Certainty of Predictions", xlabel = "Time", ylabel = "Note")
    plot.plot_data(new_data, title = "Original + Predicted Data", xlabel = "Time", ylabel = "Note")
    plot.plot_data(new_predictions, title = "Predicted Data", xlabel = "Time", ylabel = "Note")

    audio_midi.data_to_audio(new_data, "og + pred")


    

if __name__ == "__main__":
    main(sys.argv[1])