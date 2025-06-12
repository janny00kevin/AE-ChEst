import numpy as np
import matplotlib.pyplot as plt

# Set the initial learning rate and number of iterations
alpha_0 = 1e-2
iterations = 1000

# Calculate the learning rate change for both formulas
k_values = np.arange(0, iterations)
alpha_linear = alpha_0 / (1 + k_values)  # Original learning rate decay formula
beta = 0.0001
alpha_exponential = alpha_0 * np.exp(-beta * k_values)  # Exponential decay formula

# Plot the graph using linear scale
# plt.figure(figsize=(10, 6))
plt.subplot(411)
plt.plot(k_values, alpha_linear, label=r'$\alpha_\lambda^{(k)} = \frac{\alpha_\lambda^{(0)}}{1+k}$', color='blue')
plt.plot(k_values, alpha_exponential, label=r'$\alpha_\lambda^{(k)} = \alpha_\lambda^{(0)} e^{-\beta k}$', color='red')

# Set Y-axis to log scale (if needed)
# plt.yscale('log')

# Add title and labels
# plt.title('Learning Rate Decay Comparison')
plt.xlabel('Iterations (k)')
# Move x-axis label to the right
# plt.gca().xaxis.set_label_position('top')
plt.ylabel('Learning Rate')
plt.legend()

# Display the plot
plt.grid(True)
plt.savefig('learning_rate_decay_comparison.png', dpi=300, bbox_inches='tight')
