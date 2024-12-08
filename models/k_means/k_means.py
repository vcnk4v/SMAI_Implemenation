import numpy as np


class KMeans:
    def __init__(self, k, max_iters=1000, tol=1e-4, seed=42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.inertia_ = None  # To store WCSS
        self.seed = seed

    def fit(self, X):
        # Randomly initialize centroids
        np.random.seed(self.seed)
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for i in range(self.max_iters):
            # Assign clusters
            clusters = self._assign_clusters(X)

            # Calculate new centroids
            new_centroids = np.array(
                [X[clusters == j].mean(axis=0) for j in range(self.k)]
            )

            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

        # Calculate inertia (WCSS)
        self.inertia_ = self._calculate_inertia(X, clusters)

    def predict(self, X):
        # Assign clusters based on the final centroids
        return self._assign_clusters(X)

    def getCost(self):
        # Return the Within-Cluster Sum of Squares (WCSS)
        return self.inertia_

    def _assign_clusters(self, X):
        # Calculate distances from centroids and assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _calculate_inertia(self, X, clusters):
        # Calculate WCSS
        inertia = 0
        for i in range(self.k):
            inertia += np.sum((X[clusters == i] - self.centroids[i]) ** 2)
        return inertia
