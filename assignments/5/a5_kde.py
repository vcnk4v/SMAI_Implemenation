import sys
import os
from matplotlib.colors import LogNorm

sys.path.append(os.path.abspath("../../"))
from models.kde.kde import KDE
from models.gmm.gmm import GMM
import numpy as np
import matplotlib.pyplot as plt

n_large = 3000  # Number of points in the larger circle
n_small = 500  # Number of points in the smaller circle

# Generating the larger diffused circle with added noise
radius_large = 2.0
angles_large = 2 * np.pi * np.random.rand(n_large)
radii_large = radius_large * np.sqrt(np.random.rand(n_large))
x_large = radii_large * np.cos(angles_large) + 0.2 * np.random.randn(n_large)
y_large = radii_large * np.sin(angles_large) + 0.2 * np.random.randn(n_large)

# Generating the smaller dense circle with added noise
center_small = np.array([1, 1])
radius_small = 0.25
angles_small = 2 * np.pi * np.random.rand(n_small)
radii_small = radius_small * np.sqrt(np.random.rand(n_small))
x_small = (
    center_small[0]
    + radii_small * np.cos(angles_small)
    + 0.05 * np.random.randn(n_small)
)
y_small = (
    center_small[1]
    + radii_small * np.sin(angles_small)
    + 0.05 * np.random.randn(n_small)
)

# Combine the two datasets
x_combined = np.concatenate([x_large, x_small])
y_combined = np.concatenate([y_large, y_small])

# Plotting the combined dataset with noisy boundaries
plt.figure(figsize=(6, 6))
plt.scatter(x_combined, y_combined, s=1, color="black", alpha=0.7)
plt.title("Synthetic Dataset with Noisy Boundaries")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.grid()
plt.savefig("figures/synth_data.png")
# Combine the dataset into a single array
X_combined = np.column_stack((x_combined, y_combined))

# # Fit KDE model
# kde = KDE(kernel_type="gaussian", bandwidth=0.6)
# kde.fit(X_combined)

# # Visualize KDE density estimation
# kde.visualize(grid_size=100)

# data = np.vstack([x_combined, y_combined]).T
# from matplotlib.colors import LogNorm


def plot_gmm_clusters(gmm, X, ax, title):
    # Get the cluster assignments
    cluster_assignments = np.argmax(gmm.getMembership(), axis=1)

    # Plot each cluster with a different color
    for i in range(gmm.n_components):
        cluster_points = X[cluster_assignments == i]
        ax.scatter(
            cluster_points[:, 0], cluster_points[:, 1], s=15, label=f"Cluster {i+1}"
        )

    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_aspect("equal")
    ax.legend()


def plot_gmm_density(gmm, X, ax, title):
    x_min, y_min = X.min(axis=0) - 1
    x_max, y_max = X.max(axis=0) + 1
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )

    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # Calculate the density using GMM's likelihood function
    density = gmm._calculate_likelihood(grid_points).reshape(x_grid.shape)

    # Use a logarithmic color scale to handle a wide range of densities
    density[density <= 0] = 1e-10  # Avoid log(0) issues
    contour = ax.contourf(
        x_grid,
        y_grid,
        density,
        levels=20,
        cmap="viridis",
        norm=LogNorm(vmin=1e-9, vmax=0.1),
    )
    ax.scatter(X[:, 0], X[:, 1], color="red", s=5, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_aspect("equal")

    # Add a color bar as a legend
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Density (log scale)")
    cbar.ax.set_yticklabels(
        [f"$10^{{{int(np.log10(tick))}}}$" for tick in cbar.get_ticks()]
    )


fig, axs = plt.subplots(2, 2, figsize=(20, 24))

data = X_combined
# Fit and plot GMM with 2 components
gmm_2 = GMM(n_components=2)
gmm_2.fit(data)
# plot_gmm_density(gmm_2, data, axs[0][0], "GMM Density (2 Components)")
plot_gmm_clusters(gmm_2, data, axs[0][0], "GMM Clusters (2 Components)")
log_likelihood_2 = gmm_2.getLogLikelihood(data)
print(f"Log-Likelihood for GMM with 2 Components: {log_likelihood_2:.4f}")

# Fit and plot GMM with 3 components
gmm_3 = GMM(n_components=3)
gmm_3.fit(data)
# plot_gmm_density(gmm_3, data, axs[0][1], "GMM Density (3 Components)")
plot_gmm_clusters(gmm_3, data, axs[0][1], "GMM Clusters (3 Components)")
log_likelihood_3 = gmm_3.getLogLikelihood(data)
print(f"Log-Likelihood for GMM with 3 Components: {log_likelihood_3:.4f}")

# Fit and plot GMM with 4 components
gmm_4 = GMM(n_components=4)
gmm_4.fit(data)
# plot_gmm_density(gmm_4, data, axs[1][0], "GMM Density (4 Components)")
plot_gmm_clusters(gmm_4, data, axs[1][0], "GMM Clusters (4 Components)")
log_likelihood_4 = gmm_4.getLogLikelihood(data)
print(f"Log-Likelihood for GMM with 4 Components: {log_likelihood_4:.4f}")

# Fit and plot GMM with 5 components
gmm_5 = GMM(n_components=5)
gmm_5.fit(data)
# plot_gmm_density(gmm_5, data, axs[1][1], "GMM Density (5 Components)")
plot_gmm_clusters(gmm_5, data, axs[1][1], "GMM Clusters (5 Components)")

plt.tight_layout()
plt.show()
log_likelihood_5 = gmm_5.getLogLikelihood(data)
print(f"Log-Likelihood for GMM with 5 Components: {log_likelihood_5:.4f}")
