from keras import Sequential
from keras import layers
import numpy as np

def create_model(data_input_shape):
    model = Sequential()
    model.add(layers.LSTM(124, activation='relu', input_shape=(data_input_shape)))
    model.add(layers.Dense(4))

    model.compile(optimizer='rmsprop', loss='mse')
    
    return model


def train_model(model, X, y, epochs = 500):
    model.fit(X, y, epochs=epochs, verbose=1)

    return model

def predict(model, X, seq_length = 8, extension_length = 1000):
    # First predict the training data
    predictions = []

    # Then predict the extension
    x_input = X[-1]
    for i in range(extension_length):
        # Reshape x_input to (1, seq_length, features)
        x_input = x_input.reshape((1, seq_length, x_input.shape[1]))

        yhat = model.predict((x_input), verbose=0)
        yhat_ints = np.round(yhat)
        yhat_ints = yhat_ints.astype(int)
        for i in range(len(yhat_ints[0])):
            if yhat_ints[0][i] < 0:
                yhat_ints[0][i] = 0
            elif yhat_ints[0][i] > 127:
                yhat_ints[0][i] = 127
        predictions.append(yhat_ints)

        # Remove the first element of input and add the prediction to the end
        x_input = np.append(x_input[0][1:], yhat).reshape(seq_length, -1)

    return np.array(predictions)
