import numpy as np
import matplotlib.pyplot as plt

# Define a potential function
def potential(x, y):
    return np.log((x - 1)**2 + (y - 1)**2) + np.log((x + 1)**2 + (y + 1)**2)

def grad_potential(x, y, epsilon=1e-6):
    gx = (2 * (x - 1) / (((x - 1)**2 + (y - 1)**2) + epsilon)) + (2 * (x + 1) / (((x + 1)**2 + (y + 1)**2) + epsilon))
    gy = (2 * (y - 1) / (((x - 1)**2 + (y - 1)**2) + epsilon)) + (2 * (y + 1) / (((x + 1)**2 + (y + 1)**2) + epsilon))
    return -gx, -gy  # Negative gradient for "downhill" direction

# Generate grid points
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = potential(X, Y)

# Calculate gradient (vector field)
gradX, gradY = grad_potential(X, Y)

# Langevin sampling
def langevin_sampling(steps, start, lr, noise_scale):
    x, y = start
    samples = []
    for _ in range(steps):
        gx, gy = grad_potential(x, y)
        x += -lr * gx + noise_scale * np.random.randn()
        y += -lr * gy + noise_scale * np.random.randn()
        samples.append([x, y])
    return np.array(samples)

# Start Langevin sampling
samples = langevin_sampling(1000, start=[-1, -1], lr=0.01, noise_scale=0.05)

# Create the plot
plt.figure(figsize=(4,3))

# Add shaded regions for the two concentrated areas
circle1 = plt.Circle((1, 1), 0.3, color='orange', alpha=0.3)
circle2 = plt.Circle((-1, -1), 0.3, color='orange', alpha=0.3)
plt.gca().add_artist(circle1)
plt.gca().add_artist(circle2)

# Reduce density of arrows (make quiver sparser)
step = 5  # Adjust this to control sparsity
plt.quiver(X[::step, ::step], Y[::step, ::step], gradX[::step, ::step], gradY[::step, ::step], color='black')

# Remove ticks
plt.xticks([])
plt.yticks([])
plt.axis('off')
# Set axis limits
plt.xlim(-2, 2)
plt.ylim(-2, 2)

# Save the image
plt.savefig("langevin_sampling_with_shading.png", dpi=600)
plt.show()