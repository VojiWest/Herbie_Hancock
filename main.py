from Audio import audio_midi, audio_vector
from Preprocess import preprocess
from Model import LSTM_model
from Plots import plot

import pandas as pd

def data_to_audio(data, audio_name):
    cellos_midi = audio_midi.get_midi(data, instrument_num=42)
    audio_midi.write_midi(cellos_midi, name = audio_name)
    audio_midi.midi_to_wav(name = audio_name)

    return data


def main():
    data = preprocess.load_data(path = "F.txt")

    plot.plot_data(data, title = "Original Data", xlabel = "Time", ylabel = "Frequency")
    plot.plot_histogram(data, title = "Histogram of Original Data", xlabel = "Frequency", ylabel = "Count")

    data_to_audio(data, "original")

    train, val, test = preprocess.temporal_data_split(data)
    train_X, train_Y = preprocess.create_input_windows(train, seq_length = 8)

    model = LSTM_model.create_model(train_X[0].shape)
    model = LSTM_model.train_model(model, train_X, train_Y)

    predictions = LSTM_model.predict(model, train_X)

    # Reshape predictions to remove batch dimension
    print(predictions.shape)
    predictions_reshaped = predictions.reshape(predictions.shape[0], predictions.shape[2])

    # Create a DataFrame with the same column names as `train`
    new_predictions = pd.DataFrame(predictions_reshaped, columns=train.columns)
    new_data = pd.concat([train, new_predictions], ignore_index=True)

    data_to_audio(new_data, "og + pred")


    

if __name__ == "__main__":
    main()