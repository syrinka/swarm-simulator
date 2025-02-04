"""
From https://github.com/mazhar-ansari-ardeh/BenchmarkFcns
"""
# Using the matplotlib library for this example
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from evaluator import *
TARGET = Eggholder

# We want to plot the function for x and y in range [-5, 5].
# This corresponds to a grid of 100,000,000 points.
x = np.linspace(-512, 512, 1000)
y = np.linspace(-512, 512, 1000)

# `meshgrid` creates the 3D meshgrid and evaluates `ackley` on it.
# Evaluation of 100,000,000 points took less than 3 seconds.
X, Y = np.meshgrid(x, y)
Z = TARGET.infer(np.stack([X, Y]))

# Create the plot
fig = plt.figure()
ax: Axes3D = fig.add_subplot(111, projection='3d') #type:ignore

# Plot the surface and its contour with a colormap
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.contour(X, Y, Z, zdir='z', offset=0, cmap='coolwarm')

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Set view angle and display
ax.view_init(14, 120)
plt.show()
