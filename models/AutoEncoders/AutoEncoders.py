import numpy as np
import sys
import os

sys.path.append(os.path.abspath("../"))
from models.MLP.MLP import MLPClassifier


class AutoEncoder:
    def __init__(
        self,
        input_size,
        latent_dim,
        hidden_layers=[10],
        activation="sigmoid",
        optimizer="sgd",
        learning_rate=0.01,
        batch_size=32,
        epochs=50,
        early_stopping=False,
        patience=5,
        wandb_log=False,
    ):
        """
        Initializes the AutoEncoder model, including encoder and decoder.
        """

        hidden_layers = hidden_layers + [latent_dim] + hidden_layers[::-1]
        self.hidden_layers = hidden_layers
        # Encoder (reduces to latent dimensions)
        self.encoder = MLPClassifier(
            input_size=input_size,
            output_size=input_size,
            hidden_layers=hidden_layers,
            activation=activation,
            optimizer=optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping=early_stopping,
            patience=patience,
            wandb_log=wandb_log,
            task_type="regression",
            loss_function="mse",
        )

    def fit(self, X_train, X_val=None):
        """
        Trains the AutoEncoder model.
        First, the encoder is trained to reduce dimensions,
        followed by the decoder to reconstruct the original input.
        """
        # Train the encoder
        self.encoder.fit(X_train, X_train, None, None)

    def get_latent(self, X):
        """
        Passes the input data through the encoder and returns the reduced latent space representation.
        """
        activations, _ = self.encoder._forward(X)

        l_i = len(self.hidden_layers) // 2
        return activations[l_i]

    def reconstruct(self, X):
        return self.encoder.predict(X)
