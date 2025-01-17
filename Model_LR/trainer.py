def train_model(X_train_tensor, y_train_tensor, model, optimizer, criterion, num_epochs=20000):
  for epoch in range(num_epochs):
      # Forward pass
      outputs = model(X_train_tensor)
      loss = criterion(outputs, y_train_tensor)

      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if epoch % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

  print("Training finished!")

  return model


