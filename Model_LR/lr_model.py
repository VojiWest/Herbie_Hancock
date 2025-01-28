import torch
import torch.nn as nn

# Define the logistic regression model
class ActualLogisticRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActualLogisticRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.softmax(self.fc1(x), dim=1)
        return x