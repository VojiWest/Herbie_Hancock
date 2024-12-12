from keras import Sequential
from keras import layers
from keras import Model
import numpy as np
import tensorflow as tf

def create_model(data_input_shape, class_weights):
    num_classes = 31
    input_layer = layers.Input(shape=data_input_shape)

    # Define the first LSTM layer that returns sequences
    lstm_layer1 = layers.LSTM(124, activation='relu', return_sequences=True)(input_layer)

    # Define the second LSTM layer that also returns sequences
    lstm_layer2 = layers.LSTM(124, activation='relu', return_sequences=False)(lstm_layer1)
    
    # Define multiple output layers (assuming 4 outputs)
    output1 = layers.Dense(num_classes, activation='softmax', name='output1')(lstm_layer2)
    output2 = layers.Dense(num_classes, activation='softmax', name='output2')(lstm_layer2)
    output3 = layers.Dense(num_classes, activation='softmax', name='output3')(lstm_layer2)
    output4 = layers.Dense(num_classes, activation='softmax', name='output4')(lstm_layer2)

    model = Model(inputs=input_layer, outputs=[output1, output2, output3, output4])

    model.compile(optimizer='rmsprop', loss=weighted_categorical_crossentropy(class_weights))
    
    return model

def weighted_categorical_crossentropy(class_weights_list):
    # Define a custom loss function that applies class weights
    # change class_weights to a tensor from a dictionary
    
    # For each voice (output), combine the class weights
    class_weights_per_voice = []
    for cw in class_weights_list:
        # Convert each dictionary of class weights to a tensor
        class_weights = [cw.get(i, 1.0) for i in range(max(cw.keys()) + 1)]
        class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
        class_weights_per_voice.append(class_weights_tensor)
    print(class_weights_per_voice)
    
    # Now we have a list of tensors, each corresponding to the class weights for a voice
    # The length of class_weights_per_voice should be the number of outputs/voices

    def loss(y_true, y_pred):
        print("Shape yt: ", y_true)
        print("Shape yp: ", y_pred)
        print("Shape cw/v: ", len(class_weights_per_voice))
        weights = tf.reduce_sum(class_weights_tensor * y_true, axis=-1)
        print("Shape w: ", weights.shape)
        unweighted_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        print("Shape ul: ", unweighted_loss.shape)
        return unweighted_loss * weights
    return loss

def train_model(model, X, y, epochs = 250):
    # model.fit(X, y, epochs=epochs, verbose=1)
    model.fit(X, [y[:, 0], y[:, 1], y[:, 2], y[:, 3]], epochs=epochs, verbose=1)

    return model

def predict(model, X, voice_ranges, seq_length = 8, extension_length = 500):
    # First predict the training data
    predictions = []
    all_preds = []

    # Then predict the extension
    x_input = X[-1]
    for i in range(extension_length):
        # Reshape x_input to (1, seq_length, features)
        x_input = x_input.reshape((1, seq_length, x_input.shape[1]))

        yhat = model.predict((x_input), verbose=0)
        all_preds.append(yhat)
        pred_notes = []
        for i in range(4):
            pred_note = np.argmax(yhat[i])
            if pred_note == 0:
                pred_notes.append(0)
            else:
                pred_notes.append(pred_note + voice_ranges[i][0])
        predictions.append(pred_notes)

        # Remove the first element of input and add the prediction to the end
        new_note = np.array(pred_notes)
        x_input = np.concatenate((x_input[0, 1:, :], new_note.reshape(1, 4)), axis=0)

    return np.array(predictions), np.array(all_preds)
