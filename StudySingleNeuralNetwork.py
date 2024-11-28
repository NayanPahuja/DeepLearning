import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate random data
def generate_data(n_samples=100):
    class_1 = np.random.randn(n_samples, 2) + np.array([2, 2])
    class_2 = np.random.randn(n_samples, 2) + np.array([-2, -2])
    X = np.vstack((class_1, class_2))
    y = np.hstack((np.ones(n_samples), np.zeros(n_samples)))
    return torch.FloatTensor(X), torch.FloatTensor(y)

# Single-layer neural network
class SingleLayerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleLayerNN, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Function to train the model with a given learning rate
def train_model(X, y, learning_rate, epochs=1000):
    input_size = 2
    output_size = 1
    model = SingleLayerNN(input_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return model, losses

# Generate data
X, y = generate_data(100)

# Test different learning rates
learning_rates = [0.001, 0.01, 0.1, 1.0]
all_losses = {}

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    model, losses = train_model(X, y, learning_rate=lr)
    all_losses[lr] = losses

# Test the model
X_test, y_test = generate_data(100)
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    predicted = (test_outputs > 0.5).float()
    accuracy = (predicted.squeeze() == y_test).float().mean()
    print(f'Accuracy: {accuracy.item():.2f}')

# Plot loss curves for different learning rates
plt.figure(figsize=(12, 8))
for lr, losses in all_losses.items():
    plt.plot(losses, label=f'LR = {lr}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs for Different Learning Rates')
plt.legend()
plt.show()

# Visualize results for the best performing model
best_lr = min(all_losses, key=lambda x: all_losses[x][-1])
best_model, _ = train_model(X, y, learning_rate=best_lr)

# Generate new test data for visualization
X_test, y_test = generate_data(100)
with torch.no_grad():
    best_model.eval()
    test_outputs = best_model(X_test)
    predicted = (test_outputs > 0.5).float()

# Plot classification results
plt.figure(figsize=(10, 8))
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], c='red', label='Class 0')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], c='blue', label='Class 1')

# Highlight misclassified points
misclassified = X_test[predicted.squeeze() != y_test]
plt.scatter(misclassified[:, 0], misclassified[:, 1], c='green', marker='x', s=100, label='Misclassified')

plt.legend()
plt.title(f'Classification Results (Best LR = {best_lr})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
