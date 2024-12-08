import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import imageio

sys.path.append(os.path.abspath("../../"))

from performance_measures import lin_reg_metrics


class LinearRegression:
    def __init__(
        self,
        degree=1,
        regularisation_param=0,
        learning_rate=0.01,
        tolerance=1e-6,
        max_iter=1000,
        seed=42,
        regularisation_type=None,
    ):
        self.degree = degree
        self.regularisation_param = regularisation_param
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.regularisation_type = regularisation_type
        self.seed = seed
        self.weights = None

        if seed is not None:
            np.random.seed(seed)

    def _prepare_features(self, X):
        features = np.column_stack([X**i for i in range(0, self.degree + 1)])
        return features

    def _plot_figures(self, X, y, mses, varis, stds, image_path, i):
        plt.clf()

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        ax[0, 0].scatter(X, y, color="blue", label="Original Data")
        x_range = np.linspace(X.min(), X.max(), 100)
        y_line = self.predict(x_range)
        ax[0, 0].plot(x_range, y_line, color="red", label="Fitted Line")

        ax[0, 0].set_title(f"Fitting Curve at Iteration {i}")
        ax[0, 0].set_xlabel("X")
        ax[0, 0].set_ylabel("y")
        ax[0, 0].legend()

        ax[0, 1].plot(range(len(mses)), mses, label=f"MSE : {mses[-1]}")
        ax[0, 1].set_title("MSE over Iterations")
        ax[0, 1].set_xlabel("Iteration")
        ax[0, 1].set_ylabel("MSE")
        ax[0, 1].legend()

        ax[1, 0].plot(range(len(varis)), varis, label=f"Var : {varis[-1]}")
        ax[1, 0].set_title("Vars over Itrs")
        ax[1, 0].set_xlabel("Itr")
        ax[1, 0].set_ylabel("Var")
        ax[1, 0].legend()

        ax[1, 1].plot(range(len(stds)), stds, label=f"STD : {stds[-1]}")
        ax[1, 1].set_title("STD over Itrs")
        ax[1, 1].set_xlabel("Itr")
        ax[1, 1].set_ylabel("STD")
        ax[1, 1].legend()

        plt.tight_layout()

        plt.savefig(image_path)
        plt.close(fig)

    def create_gif(self, image_paths, gif_path, duration=0.5):
        images = []
        for filename in image_paths:
            images.append(imageio.imread(filename))

        imageio.mimsave(gif_path, images, duration=duration)

    def fit(self, X, y):
        X_poly = self._prepare_features(X)
        m, n = X_poly.shape
        self.weights = np.random.randn(n)

        imgs = []
        mses = []
        stds = []
        varis = []
        prev_mse = float("inf")
        convergence = 0

        for i in range(0, self.max_iter):
            predictions = X_poly @ self.weights
            errors = predictions - y
            gradient = (2 / m) * (X_poly.T @ errors)

            if self.regularisation_type == "l2":
                gradient += (2 * self.regularisation_param) * self.weights
            elif self.regularisation_type == "l1":
                gradient += self.regularisation_param * np.sign(self.weights)

            self.weights -= self.learning_rate * gradient

            metrics = lin_reg_metrics.RegressionMetrics(y, predictions)
            mse = metrics.mse()
            mses.append(mse)
            std_dev = metrics.std_deviation()
            stds.append(std_dev)
            variance = metrics.variance()
            varis.append(variance)

            convergence = i + 1
            if abs(prev_mse - mse) < self.tolerance:
                print(f"Convergence achieved at iteration {convergence}")

                break
            prev_mse = mse

            if self.degree < 6 and i % 10 == 0 and self.regularisation_type == None:

                image_path = f"figures/regression/{i}_degree{self.degree}.png"
                self._plot_figures(X, y, mses, varis, stds, image_path, i)
                imgs.append(image_path)

        if self.degree < 6 and self.regularisation_type == None and self.seed == 42:
            self.create_gif(imgs, f"figures/regression_degree_{self.degree}.gif")

        return convergence

    def predict(self, X):
        X_poly = self._prepare_features(X)
        return X_poly @ self.weights

    def save_model(self, file_path):
        np.save(file_path, self.weights)

    def load_model(self, file_path):
        self.weights = np.load(file_path)
