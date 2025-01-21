def train_model(X_train, y_train, X_val, y_val, model, optimizer, criterion, num_epochs=10000):
  for epoch in range(num_epochs):
      # Forward pass
      outputs = model(X_train)
      loss = criterion(outputs, y_train)

      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if epoch % 250 == 0:
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

  print("Training finished!")

  return model


