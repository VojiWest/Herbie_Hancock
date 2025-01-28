import matplotlib.pyplot as plt

def train_model(X_train, y_train, X_val, y_val, model, optimizer, criterion, num_epochs=25000, plot_losses=True):
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

        if plot_losses:
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

            losses.append((loss.item(), val_loss.item()))

        if epoch % 500 == 0:
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            if epoch % 5000 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

            if val_loss < min_val_loss:
                min_val_loss = val_loss
            if val_loss > 1.05 * min_val_loss:
                print("Early stopping!")
                break

    print("Training finished!")

    if plot_losses:
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend(['Train', 'Test'])
        plt.savefig("Saved Plots/Final_Loss_Plot.png")
        plt.close()
        # plt.show()

    return model


