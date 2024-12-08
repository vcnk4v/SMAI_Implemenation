import numpy as np
import matplotlib.pyplot as plt


class KDE:
    def __init__(self, kernel_type="gaussian", bandwidth=1.0):
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        self.data = data

    def kernel(self, x, xi):
        u = np.linalg.norm(x - xi, axis=-1) / self.bandwidth
        if self.kernel_type == "box":
            return 0.5 * (np.abs(u) <= 1)
        elif self.kernel_type == "gaussian":
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
        elif self.kernel_type == "triangular":
            return (1 - np.abs(u)) * (np.abs(u) <= 1)
        else:
            raise ValueError("Invalid kernel type")

    def predict(self, x):
        # Vectorized kernel calculation
        diff = self.data - x
        dist = np.linalg.norm(diff, axis=1) / self.bandwidth
        kernel_values = self.kernel(x, self.data)  # Vectorized
        return np.sum(kernel_values) / (len(self.data) * self.bandwidth ** len(x))

    def visualize(self, grid_size=100):
        if self.data.shape[1] != 2:
            raise ValueError("Visualization is only supported for 2D data.")

        x_min, y_min = self.data.min(axis=0) - 1
        x_max, y_max = self.data.max(axis=0) + 1
        x_grid, y_grid = np.meshgrid(
            np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
        )

        # Vectorized prediction for the entire grid
        points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
        density_estimates = np.array([self.predict(point) for point in points])
        density_estimates = density_estimates.reshape(x_grid.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(x_grid, y_grid, density_estimates, levels=20, cmap="viridis")
        plt.scatter(self.data[:, 0], self.data[:, 1], color="red", s=5, alpha=0.6)
        plt.title(f"KDE Density Estimation ({self.kernel_type.capitalize()} Kernel)")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.colorbar(label="Density")
        plt.show()
