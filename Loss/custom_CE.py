import torch
from torch import nn

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):
        # Input shape: (batch_size, num_classes)
        # Target shape: (batch_size, num_classes) - one-hot encoded

        # Clip the inputs to prevent numerical instability
        inputs = torch.clamp(inputs, min=1e-7, max=1 - 1e-7)

        # Calculate the loss
        loss = -torch.sum(targets * torch.log(inputs), dim=1)

        # Return the average loss over the batch
        return torch.mean(loss)