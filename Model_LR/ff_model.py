import torch
import torch.nn as nn

# Define the logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LogisticRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()  # or softmax for multi-class

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x) # or softmax
        return x