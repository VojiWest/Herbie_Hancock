import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=124, num_layers=1):
        super(GRUModel, self).__init__()

        # Number of classes and input dimensions
        self.num_classes = 31
        
        # Define the LSTM layers
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Define hidden dense layers
        self.hidden1 = nn.Linear(hidden_size, 124)
        
        # Define the output layers for each voice
        self.output1 = nn.Linear(124, self.num_classes)
        self.output2 = nn.Linear(124, self.num_classes)
        self.output3 = nn.Linear(124, self.num_classes)
        self.output4 = nn.Linear(124, self.num_classes)
        
    def forward(self, x, h0):
        # Forward pass through the first GRU layer
        gru_out1, hn = self.gru1(x, h0)
        
        # Select the last time step from the GRU output
        gru_out1_last = gru_out1[:, -1, :]

        # Forward pass through the hidden layers
        hidden_out1 = F.relu(self.hidden1(gru_out1_last)) # Maybe try with a different activation function
        
        # Generate outputs for all voices
        out1 = self.output1(hidden_out1)
        out2 = self.output2(hidden_out1)
        out3 = self.output3(hidden_out1)
        out4 = self.output4(hidden_out1)

        out1 = F.softmax(out1, dim=1)
        out2 = F.softmax(out2, dim=1)
        out3 = F.softmax(out3, dim=1)
        out4 = F.softmax(out4, dim=1)
        
        return (out1, out2, out3, out4), hn


def weighted_categorical_crossentropy(class_weights_list, weighted=True):
    def loss_fn(y_true_list, y_pred_list):
        # switch first and second dimension
        y_true_list = np.reshape(y_true_list, (y_true_list.shape[1], y_true_list.shape[0], y_true_list.shape[2]))
        
        # Initialize total loss
        total_loss = 0.0
        
        # Iterate over each output and its corresponding class weights
        for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            
            # Convert TensorFlow tensor or ensure it's a list/array
            if isinstance(class_weights_list[i], (tf.Tensor, np.ndarray)):
                class_weights = class_weights_list[i].numpy() if hasattr(class_weights_list[i], 'numpy') else class_weights_list[i]
            elif isinstance(class_weights_list[i], (list, tuple)):
                class_weights = class_weights_list[i]
            
            # Ensure class_weights is a PyTorch tensor and move to the appropriate device
            device = y_pred.device
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            
            # Create the loss function for this output
            if weighted:
                criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            else:
                criterion = nn.CrossEntropyLoss()
            loss = criterion(y_pred, y_true)
            total_loss += loss
        
        # Return the total loss
        return total_loss
    return loss_fn

def multi_output_mse_loss():
    def loss_fn(y_true_list, y_pred_list):
        total_loss = 0.0
        y_true_list = np.reshape(y_true_list, (y_true_list.shape[1], y_true_list.shape[0], y_true_list.shape[2]))

        for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            criterion = nn.MSELoss()
            loss = criterion(y_pred, y_true)
            total_loss += loss
        return total_loss
    return loss_fn

def multi_output_cosine_loss():
    def loss_fn(y_true_list, y_pred_list):
        total_loss = 0.0
        y_true_list = np.reshape(y_true_list, (y_true_list.shape[1], y_true_list.shape[0], y_true_list.shape[2]))

        for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            criterion = nn.CosineSimilarity()
            loss = criterion(y_pred, y_true)
            loss_sum = torch.sum(loss) + 1
            total_loss += loss_sum
        return total_loss
    return loss_fn


def create_model(data_input_shape, class_weights_tensor):
    model = GRUModel(data_input_shape[1])
    
    # Use RMSprop optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Start with 1e-3 learning rate
    
    # Use weighted categorical crossentropy loss (for multi-class classification)
    # loss_fn = weighted_categorical_crossentropy(class_weights_tensor, weighted=False)
    
    # Use MSE loss (for regression)
    loss_fn = multi_output_mse_loss()

    # Use Cosine loss (testing)
    # loss_fn = multi_output_cosine_loss()
    
    return model, optimizer, loss_fn

def get_voices_acc(y_true, y_pred):
    # Get the accuracy for each voice
    y_true = np.reshape(y_true, (y_true.shape[1], y_true.shape[0], y_true.shape[2]))

    acc = []
    for i in range(4):
        voice_acc = (torch.argmax(y_pred[i], dim=1) == torch.argmax(y_true[i], dim=1)).float().mean().item()
        voice_acc = round(voice_acc, 4)
        acc.append(voice_acc)
        
    return acc

def plot_pred_for_each_voice(y_true, y_pred, epoch, fig):
    # Get the accuracy for each voice
    y_true = np.reshape(y_true, (y_true.shape[1], y_true.shape[0], y_true.shape[2]))

    # Get the amount of correct predictions for each voice for each note (TO DO)


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
    # plt.draw()
    plt.savefig(f"Saved Plots/voice_predictions.png")
    # plt.pause(0.1)



def train_model(model, optimizer, loss_fn, train_X, train_y, epochs=100):
    model.train()

    # Create Dataset
    batch_size = 32
    dataset = torch.utils.data.TensorDataset(train_X, train_y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Create main figure
    plt.ion()
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    
    for epoch in range(epochs):
        all_outputs = [[] for _ in range(4)]  # Assuming the tuple has 4 tensors
        # h0 = None  # Initialize h0 as None for the first batch

        for i, (X_batch, y_batch) in enumerate(dataloader):
            # Forward pass
            batch_size = X_batch.size(0)
            # if h0 is None:
            #     h0 = torch.zeros(1, batch_size, 124) # Initialize hidden state

            # if h0.size(1) != batch_size:
            #     h0 = h0[:, :batch_size, :]

            h0 = torch.zeros(1, batch_size, 124)  # Initialize hidden state
            outputs, hn = model(X_batch, h0)

            # Update h0 for the next batch
            # h0 = hn.detach()

            for j in range(4):
                all_outputs[j].append(outputs[j])

            # Compute loss
            loss = loss_fn(y_batch, outputs)
            
            # Backward pass and optimize
            optimizer.zero_grad()  # Zero gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights

        all_outputs = [torch.cat(output_list, dim=0) for output_list in all_outputs]
        all_labels = train_y

        accuracy = get_voices_acc(all_labels, all_outputs)
        plot_pred_for_each_voice(all_labels, all_outputs, epoch + 1, fig)
        
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
    # hn = torch.zeros(1, 1, 124).to(device)  # Initialize hidden state
    for i in range(extension_length):
        # Forward pass to get the prediction
        with torch.no_grad():  # No need to track gradients for prediction
            h0 = torch.zeros(1, 1, 124).to(device)  # Initialize hidden state
            yhat, hn = model(x_input, h0)

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
