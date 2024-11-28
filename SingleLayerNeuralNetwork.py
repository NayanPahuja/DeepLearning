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

# Generate data
X, y = generate_data(100)

# Create the neural network
input_size = 2
output_size = 1
model = SingleLayerNN(input_size, output_size)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the neural network
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y.unsqueeze(1))

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Test the neural network
X_test, y_test = generate_data(100)
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    predicted = (test_outputs > 0.5).float()
    accuracy = (predicted.squeeze() == y_test).float().mean()
    print(f'Accuracy: {accuracy.item():.2f}')

# Visualize results
plt.figure(figsize=(10, 8))
plt.scatter(X_test[y_test == 0].numpy()[:, 0], X_test[y_test == 0].numpy()[:, 1], c='red', label='Class 0')
plt.scatter(X_test[y_test == 1].numpy()[:, 0], X_test[y_test == 1].numpy()[:, 1], c='blue', label='Class 1')

# Plot misclassified points
misclassified = X_test[predicted.squeeze() != y_test]
plt.scatter(misclassified[:, 0].numpy(), misclassified[:, 1].numpy(),
            c='green', marker='x', s=100, label='Misclassified')

plt.legend()
plt.title('PyTorch Single-Layer Neural Network Classification Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
