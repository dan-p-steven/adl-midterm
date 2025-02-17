import torch.nn as nn
import torch.nn.functional as F
import time

class FNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

    def fit(self, train_loader, criterion, optimizer, epochs=5):

        # Record epoch losses
        epoch_losses = []

        # Start timer
        start_time = time.time()

        for epoch in range(epochs):
            self.train()  # Set the model to training mode
            running_loss = 0.0

            for images, labels in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass: Compute predicted y by passing x to the model
                outputs = self(images)

                # Compute the loss
                loss = criterion(outputs, labels)

                # Backward pass: Compute gradients
                loss.backward()

                # Optimize the weights
                optimizer.step()

                # Accumulate the loss for reporting
                running_loss += loss.item()

            # Store the average loss for this epoch
            epoch_loss = running_loss / len(train_loader)
            epoch_losses.append(epoch_loss)  # Add the average loss of this epoch to the list

            # Print statistics for each epoch
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        # Record time
        end_time = time.time()
        training_time = end_time - start_time

        return epoch_losses, training_time
