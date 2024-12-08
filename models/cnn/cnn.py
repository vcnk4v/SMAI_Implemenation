import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        stride=1,
        dropout_rate=0.5,
        padding=0,
        task="classification",
    ):
        super(CNN, self).__init__()
        assert task in [
            "classification",
            "regression",
        ], "Task must be 'classification' or 'regression'"

        self.task = task

        # Define CNN layers
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=kernel_size, stride=stride, padding=padding
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        self._initialize_fc_in_features()

        # Define fully connected layers
        self.fc1 = nn.Linear(
            self.in_features, 128
        )  # Adjust based on final feature map size
        self.fc2 = nn.Linear(
            128, 1 if task == "regression" else 10
        )  # 1 for regression, 10 for classification
        # self.input_shape = input_shape
        # self.num_classes = num_classes

    def _initialize_fc_in_features(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            self.in_features = x.numel()

    def forward(self, x):
        # Pass through convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the output from conv layers
        x = x.view(-1, self.in_features)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output layer
        if self.task == "classification":
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)  # Log softmax for classification
        else:
            x = self.fc2(x)
            return x  # Raw output for regression
