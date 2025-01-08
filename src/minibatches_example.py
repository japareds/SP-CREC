import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Parameters
n = 100  # Image size (100x100 pixels)
n_b = 20  # Minibatch size (20x20 pixels)
B = (n // n_b) ** 2  # Total number of minibatches

# Illustrate a grid image with random pixel value
image = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        image[i, j] = np.random.random()  

# Plot the image
fig, ax = plt.subplots(figsize=(3,3))
ax.imshow(image, cmap='gray', interpolation='none')

# Draw minibatch boundaries
for i in range(0, n+1, n_b):
    for j in range(0, n+1, n_b):
        if i<n and j<n:
            rect = patches.Rectangle((j-0.5, i-0.5), n_b, n_b, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
ax.set_xticks(np.arange(0,n,20))
ax.set_yticks(np.arange(0,n,20))

plt.show()
