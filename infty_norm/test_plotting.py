import matplotlib.pyplot as plt
import numpy as np

# Define the data
SNR = [-4, -2, 0, 2, 4, 6, 8, 10]
LS_NMSE = [27.7988, 25.791, 23.8237, 21.7438, 19.8097, 17.7912, 15.846, 13.8442]
LMMSE_NMSE = [16.9139, 16.4243, 15.8395, 14.7099, 13.6158, 12.299, 10.9186, 9.3056]
LISTA_NMSE_1st = [9.2211, 9.1891, 9.7013, 9.5102, 9.2268, 9.2983, 9.5743, 9.4165]

# Create the figure and axes for the plot
# plt.figure(figsize=(10, 6)) # Set the figure size for better readability

# Plot the LS curve
# 'o' as marker for data points, '-' for a solid line, 'blue' color, and 'LS' label for the legend
plt.plot(SNR, LS_NMSE, marker='o', linestyle='-', linewidth=1, color="tab:blue", label='LS')

# Plot the LMMSE curve
# 's' as marker (square), '--' for a dashed line, 'red' color, and 'LMMSE' label
plt.plot(SNR, LMMSE_NMSE, marker='o', linestyle='-', linewidth=1, color="tab:red", label='LMMSE')

# Plot the LISTA_NMSE_1st curve
# '^' as marker (triangle), '-.' for a dash-dot line, 'green' color, and specific label
plt.plot(SNR, LISTA_NMSE_1st, marker='o', linestyle='-', linewidth=1, color="tab:orange", label='AE-Chest')

# Add title and axis labels for clarity
# plt.title('NMSE vs. SNR for Different Estimation Methods')

plt.xlabel('SNR (dB)') # X-axis label
plt.ylabel('NMSE (dB)') # Y-axis label

# Add a grid for easier data point reading
# 'True' to show grid, '--' for dashed lines, 'alpha' for transparency
plt.grid(True)

# Add a legend to identify each plotted line
plt.legend()

# Display the plot
plt.savefig('nmse_vs_snr.pdf', bbox_inches='tight') # Save the figure with high resolution
