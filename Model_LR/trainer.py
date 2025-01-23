import matplotlib.pyplot as plt

def train_model(X_train, y_train, X_val, y_val, model, optimizer, criterion, num_epochs=100000):
  losses = []
  min_val_loss = float('inf')
  for epoch in range(num_epochs):
      # Forward pass
      outputs = model(X_train)
      loss = criterion(outputs, y_train)

      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # val_outputs = model(X_val)
      # val_loss = criterion(val_outputs, y_val)

      # losses.append((loss.item(), val_loss.item()))

      if epoch % 1000 == 0:
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        if val_loss < min_val_loss:
          min_val_loss = val_loss
        if val_loss > 1.05 * min_val_loss:
          print("Early stopping!")
          break

  print("Training finished!")

  # plt.plot(losses)
  # plt.xlabel('Epoch')
  # plt.ylabel('Loss')
  # plt.title('Training Loss')
  # plt.show()

  return model


