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
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def fit(self, train_loader, X_test, y_test, loss_fn, optimizer, epochs=5):
        '''
        Training loop for neural network.

        Returns loss per epoch during training.
        '''


        # Record epoch losses
        epoch_losses = []
        epoch_accs = []

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

            # Calculate accuracies per epoch. Only used in the final train and
            # evaluation. Not used during KFCV
            if X_test.numel() != 0:

                # 1. forward pass on test x
                outputs = self.predict(X_test)

                # 2. compute acc with test y
                _, y_pred = torch.max(outputs, 1)
                correct = (y_pred == y_test).sum().item()
                acc = correct / y_test.size(0) * 100

                # 3. collect acc/epoch
                epoch_accs.append(acc)


            # Print statistics for each epoch
            print(f"\t\tEpoch [{epoch+1}/{epochs}] loss: {epoch_loss:.4f}")

        return epoch_losses, epoch_accs

    def predict(self, x):
        '''
        Evaluate data and return predictions.

        Returns predictions for data x.
        '''

        # Set to eval mode
        self.eval()

        # Tracker variables for accuracy and outputs
        with torch.no_grad():
            predict = self(x)

        return predict
