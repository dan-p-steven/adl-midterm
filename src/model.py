import torch
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

    def fit(self, train_loader, loss_fn, optimizer, epochs=5):

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
                loss = loss_fn(outputs, labels)

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

    def predict(self, test_loader):

        # Set to eval mode
        self.eval()

        # Tracker variables for accuracy and outputs
        correct, total = 0, 0
        outputs = []

        with torch.no_grad():
            for images, labels in test_loader:

                # Forward pass
                out = self(images)

                # Append outputs to running list
                outputs.append(out)

                _, predicted = torch.max(out, 1)

                # Add to total and correct counter
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total * 100
        outputs = torch.cat(outputs, dim=0)

        return outputs, accuracy
