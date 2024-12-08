import numpy as np
import wandb
import json

import pandas as pd


class MLPClassifier:
    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers=[64, 32],
        activation="relu",
        optimizer="sgd",
        learning_rate=0.01,
        batch_size=32,
        epochs=100,
        early_stopping=True,
        patience=5,
        wandb_log=True,
        task_type="classification",
        loss_function="cross_entropy",
    ):
        np.random.seed(42)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.weights, self.biases = self._initialize_parameters()
        self.loss_history = []
        self.wandb_log = wandb_log
        self.task_type = task_type
        self.loss_function = loss_function
        if self.task_type == "regression" and self.loss_function == "cross_entropy":
            self.loss_function = "mse"

    def get_params(self):
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "wandb_log": self.wandb_log,
            "task_type": self.task_type,
        }

    def _initialize_parameters(self):
        weights = []
        biases = []

        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(
                layer_sizes[i], layer_sizes[i + 1]
            ) * np.sqrt(2 / (layer_sizes[i] + layer_sizes[i + 1]))
            bias_vector = np.zeros((1, layer_sizes[i + 1]))
            weights.append(weight_matrix)
            biases.append(bias_vector)

        # for i in range(1, len(self.hidden_layers) + 2):
        #     weights.append(np.random.randn(layer_sizes[i - 1], layer_sizes[i]))
        #     biases.append(np.zeros((1, layer_sizes[i])))

        return weights, biases

    # def _forward(self, X):
    #     activations = [X]
    #     z_values = []
    #     for weight, bias in zip(self.weights, self.biases):
    #         z = np.dot(activations[-1], weight) + bias
    #         z_values.append(z)
    #         activation = self._activation(z)
    #         activations.append(activation)
    #     return activations, z_values

    def _forward(self, X):
        activations = [X]
        z_values = []
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], weight) + bias
            z_values.append(z)

            if i == len(self.weights) - 1:
                if self.task_type == "classification":
                    activation = self._softmax(z)  # Softmax for classification
                else:
                    activation = self._linear(z)  # Linear for regression
            else:
                activation = self._activation(z)
            activations.append(activation)

        return activations, z_values

    def _backward(self, activations, z_values, y_true):
        if isinstance(y_true, np.ndarray) and len(y_true.shape) > 0:  # If it's a batch
            m = y_true.shape[0]  # Number of samples in the batch
        else:  # Single data point
            m = 1
        gradients_w = []
        gradients_b = []
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]

        delta = (activations[-1] - y_true) / m

        for i in reversed(range(len(self.weights))):
            z = z_values[i]
            a = activations[i].reshape(-1, layer_sizes[i])
            dz = delta * self._activation_derivative(z)
            dw = np.dot(a.T, dz)
            db = np.sum(dz, axis=0, keepdims=True)
            gradients_w.append(dw)
            gradients_b.append(db)
            delta = np.dot(dz, self.weights[i].T)
        return gradients_w[::-1], gradients_b[::-1]

    def _sgd(self, X, y):
        for i in range(X.shape[0]):
            X_single = X[i]  # Single data point
            y_single = y[i]
            activations, z_values = self._forward(X_single)
            gradients_w, gradients_b = self._backward(activations, z_values, y_single)
            self._apply_gradients(gradients_w, gradients_b)

    def _batch_gradient_descent(self, X, y):
        # Calculate gradients based on the entire dataset
        activations, z_values = self._forward(X)

        gradients_w, gradients_b = self._backward(activations, z_values, y)
        self._apply_gradients(gradients_w, gradients_b)

    def _apply_gradients(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def _mini_batch_gradient_descent(self, X, y):
        indices = np.random.permutation(X.shape[0])
        X_shuffled, y_shuffled = X[indices], y[indices]
        for i in range(0, X.shape[0], self.batch_size):
            X_batch = X_shuffled[i : i + self.batch_size]
            y_batch = y_shuffled[i : i + self.batch_size]
            activations, z_values = self._forward(X_batch)
            gradients_w, gradients_b = self._backward(activations, z_values, y_batch)
            self._apply_gradients(gradients_w, gradients_b)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _sigmoid(self, z):
        z_clipped = np.clip(z, -709, 709)  # Clipping to the range [-709, 709]
        return 1 / (1 + np.exp(-z_clipped))

    def _tanh(self, z):
        return np.tanh(z)

    def _relu(self, z):
        return np.maximum(0, z)

    def _linear(self, z):
        return z

    def _activation(self, z):
        if self.activation == "sigmoid":
            return self._sigmoid(z)
        elif self.activation == "tanh":
            return self._tanh(z)
        elif self.activation == "relu":
            return self._relu(z)
        elif self.activation == "linear":
            return self._linear(z)

    def _activation_derivative(self, z):
        if self.activation == "sigmoid":
            return self._sigmoid(z) * (1 - self._sigmoid(z))
        elif self.activation == "tanh":
            return 1 - np.tanh(z) ** 2
        elif self.activation == "relu":
            return np.where(z > 0, 1, 0)
        elif self.activation == "linear":
            return np.ones_like(z)

    # Training the model
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        best_loss = np.inf
        patience_counter = 0

        for epoch in range(self.epochs):
            if self.optimizer == "sgd":
                self._sgd(X_train, y_train)
            elif self.optimizer == "batch":
                self._batch_gradient_descent(X_train, y_train)
            elif self.optimizer == "mini_batch":
                self._mini_batch_gradient_descent(X_train, y_train)

            train_loss = self._compute_loss(X_train, y_train)
            self.loss_history.append(train_loss)

            if X_val is not None:
                val_loss = self._compute_loss(X_val, y_val)
                if self.wandb_log:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                        }
                    )
            else:
                if self.wandb_log:
                    wandb.log({"epoch": epoch + 1, "train_loss": train_loss})

            if self.early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            if X_val is not None:
                print(
                    f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss}, Val Loss: {val_loss}"
                )
            else:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss}")

    # Predict method
    def predict(self, X):
        activations, _ = self._forward(X)
        return activations[-1]

    def _compute_loss(self, X, y):

        m = X.shape[0]  # Number of examples
        activations, _ = self._forward(X)
        y_pred = activations[-1]

        if self.loss_function == "cross_entropy":
            epsilon = 1e-15
            if y.shape != y_pred.shape:
                y = y.reshape(y_pred.shape)

            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.sum(y * np.log(y_pred)) / m  # Cross-entropy loss
        elif self.loss_function == "mse":
            loss = np.mean((y_pred - y) * (y_pred - y))  # MSE loss
        elif self.loss_function == "bce":
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / m
        return loss

    def save(self, path):
        model_params = self.get_params()

        with open(path, "w") as f:
            json.dump(model_params, f)

    def load(path):
        with open(path, "r") as f:
            model_params = json.load(f)
        model = MLPClassifier(
            input_size=model_params["input_size"],
            output_size=model_params["output_size"],
            hidden_layers=model_params["hidden_layers"],
            activation=model_params["activation"],
            optimizer=model_params["optimizer"],
            learning_rate=model_params["learning_rate"],
            batch_size=model_params["batch_size"],
            epochs=model_params["epochs"],
            early_stopping=model_params["early_stopping"],
            patience=model_params["patience"],
            wandb_log=model_params["wandb_log"],
            task_type=model_params["task_type"],
        )
        return model

    def gradient_checking(self, X_sample, y_sample, epsilon=1e-7):
        activations, zs = self._forward(X_sample)

        dW_analytical, db_analytical = self._backward(activations, zs, y_sample)

        dW_numerical = [np.zeros_like(w) for w in self.weights]
        db_numerical = [np.zeros_like(b) for b in self.biases]

        # Check gradients for weights
        for l in range(len(self.weights)):
            grad = np.zeros_like(self.weights[l])
            for i in range(self.weights[l].shape[0]):
                for j in range(self.weights[l].shape[1]):
                    self.weights[l][i][j] += epsilon
                    cost_plus = self._cost(X_sample, y_sample)
                    self.weights[l][i][j] -= 2 * epsilon
                    cost_minus = self._cost(X_sample, y_sample)
                    self.weights[l][i][j] += epsilon
                    grad[i][j] = (cost_plus - cost_minus) / (2 * epsilon)
            dW_numerical[l] = grad

        for l in range(len(self.biases)):
            grad = np.zeros_like(self.biases[l])
            for i in range(self.biases[l].shape[0]):
                self.biases[l][i] += epsilon
                cost_plus = self._cost(X_sample, y_sample)
                self.biases[l][i] -= 2 * epsilon
                cost_minus = self._cost(X_sample, y_sample)
                self.biases[l][i] += epsilon
                grad[i] = (cost_plus - cost_minus) / (2 * epsilon)
            db_numerical[l] = grad

        for l in range(len(self.weights)):
            print(f"Layer {l+1}")
            weight_diff = np.linalg.norm(
                dW_analytical[l] - dW_numerical[l]
            ) / np.linalg.norm(dW_analytical[l] + dW_numerical[l])
            bias_diff = np.linalg.norm(
                db_analytical[l] - db_numerical[l]
            ) / np.linalg.norm(db_analytical[l] + db_numerical[l])

            print(f"Weight difference: {weight_diff}")
            print(f"Bias difference: {bias_diff}")


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
)


class MultiLabelMLPClassifier(MLPClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = "bce"

    def _forward(self, X):
        activations = [X]
        z_values = []
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], weight) + bias
            z_values.append(z)

            # Apply activation for hidden layers, and sigmoid for the output layer
            if i == len(self.weights) - 1:  # Final layer (output layer)
                activation = self._sigmoid(z)  # Use sigmoid for multi-label output
            else:
                activation = self._activation(
                    z
                )  # Use chosen activation for hidden layers
            activations.append(activation)

        return activations, z_values

    def evaluate(self, X, y_true):
        """
        Evaluates the model on the provided dataset using accuracy, precision, recall, F1-score, and Hamming loss.
        """
        y_pred_proba = self.predict(X)
        y_pred = (y_pred_proba >= 0.5).astype(
            int
        )  # Convert probabilities to binary labels

        # Calculate metrics
        # accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=1)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=1)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=1)
        hamming = hamming_loss(y_true, y_pred)

        metrics = {
            "accuracy": 1 - hamming,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "hamming_loss": hamming,
        }

        return metrics

    def _mini_batch_gradient_descent(self, X, y):
        # Convert to numpy arrays if they are Pandas DataFrames
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()

        indices = np.random.permutation(X.shape[0])
        X_shuffled, y_shuffled = X[indices], y[indices]

        for i in range(0, X.shape[0], self.batch_size):
            X_batch = X_shuffled[i : i + self.batch_size]
            y_batch = y_shuffled[i : i + self.batch_size]
            activations, z_values = self._forward(X_batch)
            gradients_w, gradients_b = self._backward(activations, z_values, y_batch)
            self._apply_gradients(gradients_w, gradients_b)

    def _sgd(self, X, y):
        X_array = X.to_numpy()
        y_array = y
        for i in range(X_array.shape[0]):
            X_single = X_array[i]  # Now this works as expected
            y_single = y_array[i]
            activations, z_values = self._forward(X_single)
            gradients_w, gradients_b = self._backward(activations, z_values, y_single)
            self._apply_gradients(gradients_w, gradients_b)

    def _batch_gradient_descent(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()
        activations, z_values = self._forward(X)
        gradients_w, gradients_b = self._backward(activations, z_values, y)
        self._apply_gradients(gradients_w, gradients_b)

    def load(path):
        with open(path, "r") as f:
            model_params = json.load(f)
        model = MultiLabelMLPClassifier(
            input_size=model_params["input_size"],
            output_size=model_params["output_size"],
            hidden_layers=model_params["hidden_layers"],
            activation=model_params["activation"],
            optimizer=model_params["optimizer"],
            learning_rate=model_params["learning_rate"],
            batch_size=model_params["batch_size"],
            epochs=model_params["epochs"],
            early_stopping=model_params["early_stopping"],
            patience=model_params["patience"],
            wandb_log=model_params["wandb_log"],
        )
        return model

    def save(self, path):
        model_params = self.get_params()

        with open(path, "w") as f:
            json.dump(model_params, f)


class MLPRegression:
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        epochs=100,
        learning_rate=0.01,
        activation="relu",
        optimizer="batch",
        batch_size=32,
        wandb_log=True,
        early_stopping=True,
        patience=5,
    ):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.loss_history = []
        self.wandb_log = wandb_log
        self.early_stopping = early_stopping
        self.patience = patience

    def fit(self, X_train, Y_train, X_val, Y_val):
        self.X = X_train
        self.y = Y_train
        self.X_val = X_val
        self.y_val = Y_val
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]

        # inititalizing weights and biases
        for i in range(1, len(self.hidden_layers) + 2):
            self.weights.append(np.random.randn(layer_sizes[i - 1], layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i])))

        if self.optimizer == "mini-batch":
            batch_size = self.batch_size
        elif self.optimizer == "batch":
            batch_size = None
        elif self.optimizer == "sgd":
            batch_size = 1
        else:
            print("please enter a valid optimizer")
            return
        self.train(batch_size)

    def train(self, batch_size):
        X = self.X
        y = self.y
        epochs = self.epochs
        best_loss = np.inf
        patience_counter = 0
        for epoch in range(epochs):
            indexes = []
            if batch_size == None:
                indexes.append(np.arange(X.shape[0]))

            else:
                i = np.arange(X.shape[0])
                np.random.shuffle(i)
                for j in range(0, X.shape[0], batch_size):
                    indexes.append(i[range(j, j + batch_size)])

            for index in indexes:
                X_batch = X[index]
                y_batch = y[index]
                activations, weighted_sums = self.forward_propagation(X_batch)
                gradients = self.backward_propagation(X_batch, y_batch, activations)
                self.update_weights_and_biases(gradients, X_batch.shape[0])

            temp = self.loss()
            self.loss_history.append(temp)
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": temp[0],
                    "val_loss": temp[1],
                }
            )
            if self.early_stopping:
                if temp[1] < best_loss:
                    best_loss = temp[1]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            if epoch % 10 == 0:
                print(f"{epoch} epochs done => [train_loss,val_loss] = {temp}")

    def loss(self):
        X = self.X
        y = self.y
        train_loss = self.calculate_loss(y, self.predict(X))
        val_loss = self.calculate_loss(self.y_val, self.predict(self.X_val))

        return [train_loss, val_loss]

    def forward_propagation(self, x):
        activations = [x]
        weighted_sums = []

        for i in range(len(self.weights)):
            Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            weighted_sums.append(Z)
            if i == (len(self.weights) - 1):
                activations.append(Z)
            else:
                if self.activation == "sigmoid":
                    activations.append(1 / (1 + np.exp(-1 * Z)))
                elif self.activation == "tanh":
                    activations.append(np.tanh(Z))
                elif self.activation == "relu":
                    activations.append(np.maximum(0, Z))

        return activations, weighted_sums

    def backward_propagation(self, x, y, activations):

        dZ = [0] * len(self.weights)
        dZ[-1] = activations[-1] - y

        for i in range(len(self.weights) - 2, -1, -1):
            dZ[i] = np.dot(dZ[i + 1], self.weights[i + 1].T)
            if i == (len(self.weights) - 2):
                dZ[i] = dZ[i] * 1
            else:
                if self.activation == "sigmoid":
                    dZ[i] = dZ[i] * activations[i + 1] * (1 - activations[i + 1])
                elif self.activation == "tanh":
                    dZ[i] = dZ[i] * (1 - activations[i + 1] * activations[i + 1])
                elif self.activation == "relu":
                    dZ[i] = dZ[i] * np.where(activations[i + 1] > 0, 1, 0)

        gradients = []

        for i in range(len(self.weights)):
            gradients.append(np.dot(activations[i].T, dZ[i]))
        return gradients

    def calculate_loss(self, y, predicted):
        return np.mean((y - predicted) * (y - predicted))  # MSE

    def update_weights_and_biases(self, gradients, num_samples):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients[i] / num_samples
            self.biases[i] -= (
                self.learning_rate
                * np.sum(gradients[i], axis=0, keepdims=True)
                / num_samples
            )

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        z = np.copy(activations[-1])
        return z
