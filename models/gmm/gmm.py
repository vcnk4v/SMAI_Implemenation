import numpy as np
import copy

from scipy.stats import multivariate_normal


class GMM:
    def __init__(
        self,
        n_components,
        n_iterations=100,
        tol=1e-6,
        reg_covar=1e-8,
        random_state=42,
    ):
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.means = None
        self.covariances = None
        self.weights = None
        self.responsibilities = None

    def _initialize_parameters(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # Initialize weights uniformly
        # self.weights = np.full(self.n_components, 1 / self.n_components)

        self.weights = np.random.random(self.n_components)
        self.weights /= np.sum(self.weights)

        # Randomly choose data points as initial means
        rng = np.random.default_rng()
        random_idx = rng.choice(n_samples, self.n_components, replace=False)
        self.means = X[random_idx]

        self.covariances = np.array(
            [np.eye(n_features) for _ in range(self.n_components)]
        )

    def _e_step(self, X):
        n_samples, _ = X.shape
        self.responsibilities = np.zeros((n_samples, self.n_components))

        for c in range(self.n_components):
            distribution = multivariate_normal(
                mean=self.means[c], cov=self.covariances[c], allow_singular=True
            )
            log_prob = distribution.logpdf(X)
            self.responsibilities[:, c] = np.exp(log_prob + np.log(self.weights[c]))

        # Normalize responsibilities to sum to 1 for each sample, adding a small epsilon to prevent division by zero
        total_responsibility = np.sum(self.responsibilities, axis=1, keepdims=True)
        # total_responsibility = np.maximum(
        #     total_responsibility, 1e-10
        # )  # Prevent division by zero
        total_responsibility = np.where(
            total_responsibility == 0, 1, total_responsibility
        )
        self.responsibilities /= total_responsibility

    def _m_step(self, X):
        n_samples, n_features = X.shape

        for c in range(self.n_components):
            responsibility_c = self.responsibilities[:, c]
            total_responsibility_c = responsibility_c.sum()

            # Update weights
            self.weights[c] = total_responsibility_c / n_samples

            # Update means
            self.means[c] = (
                np.sum(responsibility_c[:, np.newaxis] * X, axis=0)
                / total_responsibility_c
            )

            # Update covariance matrices
            diff = X - self.means[c]
            weighted_cov = (
                np.dot((responsibility_c[:, np.newaxis] * diff).T, diff)
                / total_responsibility_c
            )
            self.covariances[c] = weighted_cov

    def fit(self, X):
        self._initialize_parameters(X)

        log_likelihoods = []

        for iteration in range(self.n_iterations):

            prev_means = copy.deepcopy(self.means)
            prev_covariances = copy.deepcopy(self.covariances)
            prev_weights = copy.deepcopy(self.weights)
            prev_responsibilities = copy.deepcopy(self.responsibilities)
            # E-step: Calculate responsibilities
            self._e_step(X)

            # M-step: Update parameters
            self._m_step(X)

            log_likelihood = self.getLogLikelihood(X)
            log_likelihoods.append(log_likelihood)

            # Check for convergence
            if iteration > 0 and abs(log_likelihood - log_likelihoods[-2]) < self.tol:
                break

            if (
                len(log_likelihoods) > 1
                and log_likelihood < log_likelihoods[-2]
                and log_likelihoods[-2] - log_likelihood > 10
            ):
                self.means = prev_means
                self.covariances = prev_covariances
                self.weights = prev_weights
                self.responsibilities = prev_responsibilities
                break

        self.log_likelihood_ = log_likelihood

    def getParams(self):
        return {
            "weights": self.weights,
            "means": self.means,
            "covariances": self.covariances,
        }

    def getMembership(self):
        return self.responsibilities

    def _calculate_likelihood(self, X):
        # Helper function to calculate the total likelihood for each component
        total_likelihood = np.zeros(X.shape[0])

        for k in range(self.n_components):
            dist = multivariate_normal(
                mean=self.means[k], cov=self.covariances[k], allow_singular=True
            )
            total_likelihood += self.weights[k] * dist.pdf(X)

        return total_likelihood

    def getLogLikelihood(self, X):
        # Returns the average log-likelihood of the data given the model
        total_likelihood = self._calculate_likelihood(X)
        return np.mean(np.log(total_likelihood))

    def getLikelihood(self, X):
        # Returns the average likelihood of the data given the model
        total_likelihood = self._calculate_likelihood(X)
        return np.mean(total_likelihood)

    def _calculate_num_params(self, n_features):
        """Calculate the number of parameters in the GMM."""
        # Number of parameters: means + covariances + weights
        cov_params = n_features * (n_features + 1) / 2
        num_params = self.n_components * (n_features + cov_params) + (
            self.n_components - 1
        )
        return num_params

    def getAIC(self, X):
        n_samples, n_features = X.shape
        log_likelihood = self.getLogLikelihood(X)
        num_params = self._calculate_num_params(n_features)

        # AIC formula
        aic = 2 * num_params - 2 * log_likelihood
        return aic

    def getBIC(self, X):
        n_samples, n_features = X.shape
        log_likelihood = self.getLogLikelihood(X)
        num_params = self._calculate_num_params(n_features)

        # BIC formula
        bic = np.log(n_samples) * num_params - 2 * log_likelihood
        return bic
