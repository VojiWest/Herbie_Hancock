import torch

def evaluate_model(model, X_test, y_test):
  X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
  y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

  criterion = CustomCrossEntropyLoss()

  # Evaluate the model on the test set
  with torch.no_grad():
      outputs = model(X_test_tensor)
      loss = criterion(outputs, y_test_tensor)
      _, predicted = torch.max(outputs, 1)
      _, actual = torch.max(y_test_tensor, 1)
      correct_predictions = (predicted == actual).sum().item()
      accuracy = (correct_predictions / len(y_test)) * 100
      print(f'Test Loss: {loss.item():.4f}')
      print(f'Test Accuracy: {accuracy:.2f}%')
