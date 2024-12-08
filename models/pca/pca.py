import numpy as np


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X):
        # Center the data by subtracting the mean
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Calculate the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        sorted_eigenvalues = eigenvalues[sorted_indices]

        # Store the explained variance (eigenvalues)
        self.explained_variance = sorted_eigenvalues

        # Select the top n_components eigenvectors
        if self.n_components is not None:
            self.components = sorted_eigenvectors[:, : self.n_components]
        else:
            self.components = sorted_eigenvectors

    def transform(self, X):
        # Project the data onto the principal components
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def checkPCA(self, X):
        # Check if the reduced dimensions match n_components
        transformed_data = self.transform(X)
        if transformed_data.shape[1] != self.n_components:
            return False

        X_recieved = self.inverse_transform(transformed_data)
        return np.allclose(X, X_recieved, atol=5, equal_nan=False)

    def inverse_transform(self, X_transformed):
        # Transform the reduced dimension data back to the original spac
        return np.dot(X_transformed, self.components.T) + self.mean
