import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Split the training dataset into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, use_dropout=False):
        super(SimpleNN, self).__init__()
        self.use_dropout = use_dropout
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        x = torch.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc3(x)
        return x

# Create model instances with and without dropout
model_no_dropout = SimpleNN(use_dropout=False).to(device)
model_with_dropout = SimpleNN(use_dropout=True).to(device)

# Define loss function and optimizers
criterion = nn.CrossEntropyLoss()
optimizer_no_dropout = optim.Adam(model_no_dropout.parameters(), lr=0.001)
optimizer_with_dropout = optim.Adam(model_with_dropout.parameters(), lr=0.001)

# Training function
def train(model, optimizer, train_loader, val_loader):
    model.train()
    train_acc, val_acc = [], []

    for epoch in range(5):
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc.append(100 * correct / total)

        # Validation
        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc.append(100 * correct / total)
        model.train()

        print(f'Epoch {epoch + 1}, Train Acc: {train_acc[-1]:.2f}%, Val Acc: {val_acc[-1]:.2f}%')

    return train_acc, val_acc

# Train models
print("Training model without dropout:")
train_acc_no_dropout, val_acc_no_dropout = train(model_no_dropout, optimizer_no_dropout, train_loader, val_loader)

print("\nTraining model with dropout:")
train_acc_with_dropout, val_acc_with_dropout = train(model_with_dropout, optimizer_with_dropout, train_loader, val_loader)

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Evaluate models
test_acc_no_dropout = evaluate(model_no_dropout, test_loader)
test_acc_with_dropout = evaluate(model_with_dropout, test_loader)

print(f"\nTest Accuracy w/o Dropout: {test_acc_no_dropout:.2f}%")
print(f"Test Accuracy w/ Dropout: {test_acc_with_dropout:.2f}%")

# Plot training and validation accuracy
plt.figure(figsize=(12, 5))

# Plot model without dropout
plt.subplot(1, 2, 1)
plt.plot(train_acc_no_dropout, label='Train Accuracy')
plt.plot(val_acc_no_dropout, label='Validation Accuracy')
plt.title("Model without Dropout")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plot model with dropout
plt.subplot(1, 2, 2)
plt.plot(train_acc_with_dropout, label='Train Accuracy')
plt.plot(val_acc_with_dropout, label='Validation Accuracy')
plt.title("Model with Dropout")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
