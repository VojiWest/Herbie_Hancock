from Model import ff_model, lr_model, trainer, evaluator
from Loss import custom_CE
from Utils import utils
from Dataset import dataset
import torch
import torch.optim as optim
import sys

def main(mode):
    window_size=64
    ds = dataset.CustomDataset(window_size=window_size)
    # Hyperparameters
    input_size = window_size  # window size
    hidden_size = 64 # Example hidden size
    output_size = ds.get_unique_target_values() # Number of unique notes
    learning_rate = 0.001

    # Initialize the model, loss function, and optimizer
    model = lr_model.ActualLogisticRegressionModel(input_size, output_size)
    criterion = custom_CE.CustomCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_train, y_train, X_test, y_test = ds.get_train_test()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    trainer.train_model(X_train_tensor, y_train_tensor)
    evaluator.evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main(sys.argv[1])
