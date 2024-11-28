import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import logging

# Configure logging
logging.basicConfig(filename='training_logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds for reproducibility
torch.manual_seed(58)
np.random.seed(58)

# Load Iris dataset
iris = sns.load_dataset('iris')

# Prepare the data
X = iris.drop('species', axis=1).values
y = pd.get_dummies(iris['species']).values

# Initialize lists for losses and accuracies at different split sizes
split_sizes = [0.3, 0.6, 0.9]
split_train_losses = []
split_val_losses = []
split_train_accuracies = []
split_val_accuracies = []

# Loop over different training/test set sizes
for t_size in split_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=42)
    logging.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    # Define the neural network model
    class IrisNet(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(IrisNet, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Create and train the model
    num_features = X_train.shape[1]
    num_classes = y_train.shape[1]
    model = IrisNet(num_features, num_classes)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    epochs = 50
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Training phase
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = correct / total

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
            _, val_predicted = torch.max(val_outputs.data, 1)
            _, val_labels = torch.max(y_test.data, 1)
            val_accuracy = (val_predicted == val_labels).float().mean().item()

        train_losses.append(train_loss)
        val_losses.append(val_loss.item())
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    # Store results for different splits
    split_train_losses.append(train_losses)
    split_val_losses.append(val_losses)
    split_train_accuracies.append(train_accuracies)
    split_val_accuracies.append(val_accuracies)

# Plot loss comparison
plt.figure(figsize=(14, 12))
for i, t_size in enumerate(split_sizes):
    plt.plot(split_train_losses[i], label=f'Train Loss (Split: {1 - t_size})')
    plt.plot(split_val_losses[i], label=f'Val Loss (Split: {1 - t_size})', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Comparison')
plt.legend()

# Plot accuracy comparison
plt.figure(figsize=(14, 12))
for i, t_size in enumerate(split_sizes):
    plt.plot(split_train_accuracies[i], label=f'Train Acc (Split: {1 - t_size})')
    plt.plot(split_val_accuracies[i], label=f'Val Acc (Split: {1 - t_size})', linestyle='--')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend()

# Display plots
plt.show()
