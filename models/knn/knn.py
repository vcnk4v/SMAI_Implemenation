import numpy as np
import time


class KNN:
    def __init__(self, k=3, distance_metric="manhattan", optimization_type="optimized"):
        self.k = k
        self.distance_metric = distance_metric
        self.optimization_type = optimization_type

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        start_time = time.time()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions = [self._predict(x) for x in X]
        end_time = time.time()
        inference_time = end_time - start_time

        return {"predictions": np.array(predictions), "inference_time": inference_time}

    def _predict(self, x):
        if self.optimization_type != "optimized":
            distances = self._compute_distances_naive(x)
        else:
            distances = self._compute_distances(x)
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return self._most_common_label(k_nearest_labels)

    def _compute_distances(self, x):
        if self.distance_metric == "euclidean":
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            return distances
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(self.X_train - x), axis=1)
        elif self.distance_metric == "cosine":
            dot_product = np.dot(self.X_train, x)
            x_norm = np.linalg.norm(x)
            X_train_norm = np.linalg.norm(self.X_train, axis=1)
            cosine_similarity = dot_product / (x_norm * X_train_norm)
            cosine_distance = 1 - cosine_similarity
            return cosine_distance
        else:
            raise ValueError("Unsupported distance metric")

    def _compute_distances_naive(self, x):
        distances = []
        for train_x in self.X_train:
            if self.distance_metric == "euclidean":
                squared_sum = 0
                for i in range(len(train_x)):
                    squared_sum += (train_x[i] - x[i]) ** 2
                distance = squared_sum**0.5  # equivalent to np.sqrt(squared_sum)

            elif self.distance_metric == "manhattan":
                abs_sum = 0
                for i in range(len(train_x)):
                    abs_sum += abs(
                        train_x[i] - x[i]
                    )  # equivalent to np.sum(np.abs(train_x - x))
                distance = abs_sum

            elif self.distance_metric == "cosine":
                dot_product = 0
                train_x_norm = 0
                x_norm = 0
                for i in range(len(train_x)):
                    dot_product += train_x[i] * x[i]
                    train_x_norm += train_x[i] ** 2
                    x_norm += x[i] ** 2
                cosine_similarity = dot_product / ((train_x_norm**0.5) * (x_norm**0.5))
                distance = 1 - cosine_similarity

            else:
                raise ValueError("Unsupported distance metric")

            distances.append(distance)

        return distances

    def _most_common_label(self, labels):
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]
