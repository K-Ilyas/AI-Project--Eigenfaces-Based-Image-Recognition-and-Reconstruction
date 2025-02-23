import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D grid (x, y, z) and flatten it to a 1D array
width, height, depth = 4, 4, 4
x, y, z = np.meshgrid(np.arange(width), np.arange(height), np.arange(depth))

# Create a flattened index array corresponding to the 3D grid
index = z * (width * height) + y * width + x

# Create a plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D grid points
ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=index.flatten(), cmap='viridis', s=100)

# Annotate the points with their corresponding 1D index values
for i in range(x.size):
    ax.text(x.flatten()[i], y.flatten()[i], z.flatten()[i], f'{index.flatten()[i]}', color='black', fontsize=10)

# Set labels for x, y, and z axes
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Set title
ax.set_title('3D Grid Flattened into 1D Array with Index Calculation')

plt.show()
