import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MultiOutputLSTM(nn.Module):
    def __init__(self, data_input_shape, class_weights):
        super(MultiOutputLSTM, self).__init__()

        # Number of classes and input dimensions
        self.num_classes = 31
        self.input_dim = data_input_shape[1]
        
        # Define the LSTM layers
        self.gru1 = nn.GRU(input_size=self.input_dim, hidden_size=124, num_layers=3, batch_first=True, dropout=0.5)
        self.gru2 = nn.GRU(input_size=124, hidden_size=124, num_layers=3, batch_first=True, dropout=0.5)

        # Define hidden dense layers
        self.hidden1 = nn.Linear(124, 124)
        self.hidden2 = nn.Linear(124, 124)
        
        # Define the output layers for each voice
        self.output1 = nn.Linear(124, self.num_classes)
        self.output2 = nn.Linear(124, self.num_classes)
        self.output3 = nn.Linear(124, self.num_classes)
        self.output4 = nn.Linear(124, self.num_classes)
        
        # Class weights for the loss function
        # self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
    def forward(self, x):
        # Forward pass through the first LSTM layer
        gru_out1, _ = self.gru1(x)
        
        # Forward pass through the second LSTM layer
        gru_out2, _ = self.gru2(gru_out1)
        
        # Select the last time step from the LSTM output
        gru_out2_last = gru_out2[:, -1, :]

        # Forward pass through the hidden layers
        hidden_out1 = F.relu(self.hidden1(gru_out2_last))
        hidden_out2 = F.relu(self.hidden2(hidden_out1))
        
        # Generate outputs for all voices
        out1 = self.output1(hidden_out2)
        out2 = self.output2(hidden_out2)
        out3 = self.output3(hidden_out2)
        out4 = self.output4(hidden_out2)

        out1 = F.softmax(out1, dim=1)
        out2 = F.softmax(out2, dim=1)
        out3 = F.softmax(out3, dim=1)
        out4 = F.softmax(out4, dim=1)
        
        return out1, out2, out3, out4


def weighted_categorical_crossentropy(class_weights_list):
    def loss_fn(y_true_list, y_pred_list):
        # switch first and second dimension
        y_true_list = np.swapaxes(y_true_list, 0, 1)
        # Ensure y_true_list and y_pred_list have the same length
        if len(y_true_list) != len(y_pred_list):
            raise ValueError("y_true_list and y_pred_list must have the same length.")
        
        # Initialize total loss
        total_loss = 0.0
        
        # Iterate over each output and its corresponding class weights
        for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            # print(f"Shape of y_true[{i}]: ", y_true.shape)
            # print(f"Shape of y_pred[{i}]: ", y_pred.shape)
            # print(f"Length of class_weights_list: {len(class_weights_list)}")
            # print(f"class_weights_list[{i}]: {class_weights_list[i]}")

            # print predictions and true values
            # print("y_true: ", y_true[0])
            # print("y_pred: ", y_pred[0])
            
            # Convert TensorFlow tensor or ensure it's a list/array
            if isinstance(class_weights_list[i], (tf.Tensor, np.ndarray)):
                class_weights = class_weights_list[i].numpy() if hasattr(class_weights_list[i], 'numpy') else class_weights_list[i]
            elif isinstance(class_weights_list[i], (list, tuple)):
                class_weights = class_weights_list[i]
            else:
                raise TypeError(f"Unsupported type for class_weights_list[{i}]: {type(class_weights_list[i])}")
            # print(f"class_weights: {class_weights}")
            
            # Ensure class_weights is a PyTorch tensor and move to the appropriate device
            device = y_pred.device
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            
            # Create the loss function for this output
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            # criterion = nn.CrossEntropyLoss()
            loss = criterion(y_pred, y_true)
            total_loss += loss
        
        # Return the total loss
        return total_loss
    return loss_fn


def create_model(data_input_shape, class_weights):
    # Convert class_weights to a tensor
    # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    class_weights_tensor = class_weights

    model = MultiOutputLSTM(data_input_shape, class_weights_tensor)
    
    # Use RMSprop optimizer
    optimizer = optim.RMSprop(model.parameters())
    
    # Define loss function
    loss_fn = weighted_categorical_crossentropy(class_weights_tensor)
    
    return model, optimizer, loss_fn

def get_voices_acc(y_true, y_pred):
    # Get the accuracy for each voice
    y_true = np.swapaxes(y_true, 0, 1)

    acc = []
    for i in range(4):
        voice_acc = (torch.argmax(y_pred[i], dim=1) == torch.argmax(y_true[i], dim=1)).float().mean().item()
        # for pred in range(1):
        #     # print the first 5 predictions
        #     print("Pred Voice", i + 1, ":", y_pred[i][pred])
        #     print("True Voice", i + 1, ":", y_true[i][pred])
        voice_acc = round(voice_acc, 4)
        acc.append(voice_acc)
        
    return acc

def plot_pred_for_each_voice(y_true, y_pred, epoch, fig):
    # Get the accuracy for each voice
    y_true = np.swapaxes(y_true, 0, 1)

    fig.clf()

    for i in range(4):
        preds = torch.argmax(y_pred[i], dim=1)
        trues = torch.argmax(y_true[i], dim=1)

        one_to_31 = torch.tensor(range(1, 32))
        preds = torch.cat((preds, one_to_31), dim=0)

        # plot histogram of predictions in subplot
        plt.subplot(2, 2, i + 1)
        plt.hist(trues.cpu().numpy(), bins=31, alpha=0.5, label='True', color='g')
        plt.hist(preds.cpu().numpy(), bins=31, alpha=0.5, label='Predictions', color='r')
        plt.legend(loc='upper right')
        plt.title(f"Voice {i + 1} Predictions -- Epoch {epoch}")
    plt.draw()
    plt.savefig(f"Saved Plots/voice_predictions.png")
    plt.pause(0.1)



def train_model(model, optimizer, loss_fn, train_X, train_y, epochs=100):
    model.train()

    # Create main figure
    plt.ion()
    fig = plt.gcf()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(train_X)
        
        # Compute loss
        loss = loss_fn(train_y, outputs)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        accuracy = get_voices_acc(train_y, outputs)
        plot_pred_for_each_voice(train_y, outputs, epoch + 1, fig)
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Acc: {accuracy}")

    plt.ioff()  # Turn off interactive mode when training ends
    plt.show()  # Keep the final plot visible

    return model

def predict(model, X, voice_ranges, seq_length=8, extension_length=500, device='cpu'):
    # Prepare model for evaluation (turn off dropout, etc.)
    model.eval()
    
    # Convert X to a tensor and move to the correct device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    predictions = []
    all_preds = []

    # Then predict the extension
    x_input = X_tensor[-1].unsqueeze(0)  # Reshape to (1, seq_length, features)
    for i in range(extension_length):
        # Forward pass to get the prediction
        with torch.no_grad():  # No need to track gradients for prediction
            yhat = model(x_input)

        # Convert output to numpy for further processing
        yhat_np = [yhat[i].cpu().numpy() for i in range(4)]  # For each voice

        all_preds.append(yhat_np)
        
        pred_notes = []
        for i in range(4):
            pred_note = np.argmax(yhat_np[i])
            if pred_note == 0:
                pred_notes.append(0)
            else:
                pred_notes.append(pred_note + voice_ranges[i][0])

        predictions.append(pred_notes)

        # Update x_input: remove the first element and append the new prediction
        new_note = np.array(pred_notes)
        x_input = torch.cat((x_input[:, 1:, :], torch.tensor(new_note, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)), dim=1)

    return np.array(predictions), np.array(all_preds)
