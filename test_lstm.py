import numpy as np
from keras import Sequential
from keras import layers

# Create test data
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
X = []
y = []

seq_length = 3
for i in range(len(data) - seq_length):
    X.append(data[i:i + seq_length])
    y.append(data[i + seq_length])

X = np.array(X)
y = np.array(y)

print(X.shape, y.shape)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Create LSTM model
model = Sequential()
model.add(layers.LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mse')
print(model.summary())

model.fit(X, y, epochs=1000, verbose=0)

# Test the model
x_input = np.array([1, 2, 3])
x_input = np.reshape(x_input, (1, seq_length, 1))

for i in range(20):
    yhat = model.predict(x_input, verbose=0)
    print("Predicted:", yhat[0, 0], "Expected:", i + 4)
    output_rounded = round(yhat[0, 0])
    x_input = np.array([x_input[0, 1, 0], x_input[0, 2, 0], output_rounded])
    x_input = np.reshape(x_input, (1, seq_length, 1))

