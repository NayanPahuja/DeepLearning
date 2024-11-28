import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Define the function and its derivative
x_sym = sp.Symbol('x')
func = 3 * x_sym ** 2 - 3 * x_sym + 4
df_sym = sp.diff(func, x_sym)

# Convert symbolic functions to numerical functions
f = sp.lambdify(x_sym, func, "numpy")
df = sp.lambdify(x_sym, df_sym, "numpy")

# Gradient Descent Function
def gradient_descent(learning_rate, num_iterations):
    x = 0  # Initial guess
    xs = [x]
    ys = [f(x)]
    for _ in range(num_iterations):
        x = x - learning_rate * df(x)  # Update x using the gradient
        xs.append(x)
        ys.append(f(x))
    return xs, ys

# Parameters for Gradient Descent
lr = 0.1
iterations = 50

# Perform Gradient Descent
xs, ys = gradient_descent(lr, iterations)

# Theoretical minimum (derivative = 0)
x_min_theoretical = sp.solve(df_sym, x_sym)[0]
y_min_theoretical = f(x_min_theoretical)

# Plot the function and gradient descent path
x_plot = np.linspace(-1, 2, 100)
y_plot = f(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'b-', label='f(x)')  # Plot the function
plt.plot(xs, ys, 'ro-', label='Gradient Descent')  # Plot gradient descent path
plt.plot(x_min_theoretical, y_min_theoretical, 'go', markersize=10, label='Theoretical Minimum')  # Plot theoretical minimum
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent for f(x) = 3x^2 - 3x + 4')
plt.legend()
plt.grid(True)
plt.show()

# Print results
print(f"Gradient Descent Minimum: x = {xs[-1]:.6f}, f(x) = {ys[-1]:.6f}")
print(f"Theoretical Minimum: x = {x_min_theoretical:.6f}, f(x) = {y_min_theoretical:.6f}")
