import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from itertools import product
import seaborn as sns
import wandb
import sys
import os

sys.path.append(os.path.abspath("../../"))
from models.MLP.MLP import MLPClassifier, MultiLabelMLPClassifier
from models.AutoEncoders.AutoEncoders import AutoEncoder
from models.knn.knn import KNN

# from models.MLP.MultiLabelMLP import MultiLabelMLPClassifier


def wine_data():
    df = pd.read_csv("../../data/external/WineQT.csv")
    stats = df.describe().loc[["mean", "std", "min", "max"]]
    for column in stats.columns:
        print(
            f"{column}:\nMean: {stats[column]['mean']}\nStd: {stats[column]['std']}\nMin: {stats[column]['min']}\nMax: {stats[column]['max']}\n"
        )

    plt.figure(figsize=(15, 12))

    attributes = df.columns.drop(["Id"])
    for i, col in enumerate(attributes):
        plt.subplot(4, 3, i + 1)
        sns.histplot(df[col], kde=True, color="blue", edgecolor="black")
        plt.title(col)

    plt.tight_layout()
    plt.savefig("figures/wine_data.png")

    attributes = df.columns.drop(["Id", "quality"])

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))
    fig.subplots_adjust(hspace=0.5)

    for i, attribute in enumerate(attributes):
        row, col = divmod(i, 3)
        sns.boxplot(x="quality", y=attribute, data=df, ax=axes[row, col])
        axes[row, col].set_title(f"Quality vs {attribute}")

    if len(attributes) % 3 != 0:
        fig.delaxes(axes[-1, -1])

    plt.tight_layout()
    plt.savefig("figures/quality_vs_attributes.png")

    df.fillna(df.mean(), inplace=True)
    features = df.drop(columns=["Id", "quality"])

    standard_scaler = StandardScaler()
    standardized_data = standard_scaler.fit_transform(features)
    standardized_df = pd.DataFrame(standardized_data, columns=features.columns)
    standardized_df["Id"] = df["Id"]
    standardized_df["quality"] = df["quality"]

    print(standardized_df.head())

    plt.figure(figsize=(12, 6))
    corr = standardized_df.corr()
    sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
    plt.title("Correlation Matrix")
    plt.savefig("figures/correlation_matrix.png")

    return standardized_df


# wine_dataset = wine_data()
# with open("../../data/interim/3/wine_dataset.csv", "w") as f:
#     wine_dataset.to_csv(f, index=False)
with open("../../data/interim/3/wine_dataset.csv", "r") as f:
    wine_dataset = pd.read_csv(f)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    hamming_loss,
)


X = wine_dataset.drop(
    columns=["Id", "quality"]
).values  # Use all columns except 'Id' and 'quality'
y = wine_dataset["quality"].values  # The 'quality' column is the target

# Split into training and test sets (80% training, 10% testing, 10% validation)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)

n_classes = len(np.unique(y_train))


def one_hot_encode(y, num_classes, a=3):
    return np.eye(num_classes)[y - a]  # Shift by 3 since quality labels start at 3


# One-hot encode the labels for training and testing
y_train_oh = one_hot_encode(y_train, n_classes)
y_test_oh = one_hot_encode(y_test, n_classes)
y_val_oh = one_hot_encode(y_val, n_classes)

# config = {
#         "learning_rate": 0.01,
#         "epochs": 50,
#         "batch_size": 16,
#         "hidden_layers": [64, 32],
#         "activation": "relu",
#         "optimizer": "mini_batch",
#     }
# mlp = MLPClassifier(
#     input_size=X_train.shape[1],
#     output_size=len(np.unique(y_train)),
#     hidden_layers=config["hidden_layers"],
#     activation=config["activation"],
#     optimizer=config["optimizer"],
#     learning_rate=config["learning_rate"],
#     batch_size=config["batch_size"],
#     epochs=config["epochs"],
#     early_stopping=True,
#     patience=10,
# )


def first_mlp_run():

    wandb.init(
        project="MLP-singleeee-classifier",
        config={
            "learning_rate": 0.1,
            "epochs": 100,
            "batch_size": 32,
            "hidden_layers": [16],
            "activation": "relu",
            "optimizer": "sgd",
        },
    )

    # Access config hyperparameters using wandb.config
    config = wandb.config
    # config = {
    #     "learning_rate": 0.1,
    #     "epochs": 100,
    #     "batch_size": 32,
    #     "hidden_layers": [16],
    #     "activation": "relu",
    #     "optimizer": "sgd",
    # }
    mlp = MLPClassifier(
        input_size=X_train.shape[1],
        output_size=len(np.unique(y_train)),
        hidden_layers=config.hidden_layers,
        activation=config.activation,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        epochs=config.epochs,
        early_stopping=True,
        patience=5,
        wandb_log=False,
    )
    # mlp = MLPClassifier(
    #     input_size=X_train.shape[1],
    #     output_size=len(np.unique(y_train)),
    #     hidden_layers=config["hidden_layers"],
    #     activation=config["activation"],
    #     optimizer=config["optimizer"],
    #     learning_rate=config["learning_rate"],
    #     batch_size=config["batch_size"],
    #     epochs=config["epochs"],
    #     early_stopping=True,
    #     patience=5,
    #     wandb_log=False,
    # )

    # print(mlp.check_gradients(X_train[:10], y_train_oh[:10]))

    # Train the model
    mlp.fit(X_train, y_train_oh, X_val=X_val, y_val=y_val_oh)

    # Predict on the test set (probabilities)
    y_pred_proba = mlp.predict(X_test)

    # Convert predicted probabilities back to class labels
    y_pred = (
        np.argmax(y_pred_proba, axis=1) + 3
    )  # Adding 3 because we subtracted 3 earlier in one-hot encoding

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")

    # Log final test metrics to W&B
    wandb.log(
        {
            "test_accuracy": accuracy,
            "test_f1_score": f1,
            "test_precision": precision,
            "test_recall": recall,
        }
    )
    # Detailed classification report
    print(classification_report(y_test, y_pred))
    plt.figure(figsize=(10, 6))
    plt.plot(mlp.loss_history)
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


# first_mlp_run()


def hyperparameter_tuning_sweep():
    sweep_config = {
        "method": "grid",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "layers": {"values": [[16], [64, 32], [16, 32, 16]]},
            "learning_rate": {"values": [0.01, 0.1]},
            "epochs": {"values": [100, 1000]},
            "batch_size": {"values": [32, 128]},
            "activation": {"values": ["relu", "tanh", "sigmoid"]},
            "optimizer": {"values": ["sgd", "batch", "mini_batch"]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="MLP-HyperparameterTuningSingle")

    num_classes = len(np.unique(y_train))

    def train_model():
        wandb.init()
        config = wandb.config

        print(f"Running with config: {config}")
        wandb.run.name = f"activation={config['activation']}-optimizer={config['optimizer']}-layers={config['layers']}-lr={config['learning_rate']}-epochs={config['epochs']}-batch_size={config['batch_size']}"

        model = MLPClassifier(
            input_size=X_train.shape[1],
            output_size=num_classes,
            hidden_layers=config["layers"],
            activation=config["activation"],
            optimizer=config["optimizer"],
            learning_rate=config["learning_rate"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            early_stopping=True,
            patience=5,
            wandb_log=True,
        )
        model.fit(X_train, y_train_oh, X_val, y_val_oh)

        val_predictions_prob = model.predict(X_val)
        val_predictions = np.argmax(val_predictions_prob, axis=1) + 3
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_f1 = f1_score(y_val, val_predictions, average="macro")
        val_precision = precision_score(y_val, val_predictions, average="macro")
        val_recall = recall_score(y_val, val_predictions, average="macro")

        wandb.log(
            {
                "accuracy": val_accuracy,
                "precision": val_precision,
                "recall": val_recall,
                "f1": val_f1,
            }
        )
        wandb.finish()

    wandb.agent(sweep_id, function=train_model)


# hyperparameter_tuning_sweep()


def hyperparameter_tuning_sweep_multi(X_train, y_train, X_val, y_val):
    sweep_config = {
        "method": "grid",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "layers": {"values": [[16], [64, 32], [16, 32, 16]]},
            "learning_rate": {"values": [0.01, 0.1]},
            "epochs": {"values": [100, 1000]},
            "batch_size": {"values": [32, 128]},
            "activation": {"values": ["relu", "tanh", "sigmoid"]},
            "optimizer": {"values": ["sgd", "batch", "mini_batch"]},
        },
    }

    sweep_id = wandb.sweep(
        sweep_config, project="MLP-Hyperparameter-Tuning-Multi_Label"
    )

    def train_model():
        wandb.init()
        config = wandb.config

        print(f"Running with config: {config}")
        wandb.run.name = f"activation={config['activation']}-optimizer={config['optimizer']}-layers={config['layers']}-lr={config['learning_rate']}-epochs={config['epochs']}-batch_size={config['batch_size']}"

        model = MultiLabelMLPClassifier(
            input_size=X_train.shape[1],
            output_size=y_train.shape[1],
            hidden_layers=config["layers"],
            activation=config["activation"],
            optimizer=config["optimizer"],
            learning_rate=config["learning_rate"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            early_stopping=True,
            patience=10,
            wandb_log=True,
        )
        model.fit(X_train, y_train, X_val, y_val)

        metrics = model.evaluate(X_val, y_val)

        wandb.log(
            {
                "accuracy": 1 - metrics["hamming_loss"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1_score"],
                "hamming_loss": metrics["hamming_loss"],
            }
        )
        wandb.finish()

    wandb.agent(sweep_id, function=train_model)


def evaluate_best_model_on_test():
    # best_model = MLPClassifier.load("best_model.pkl")
    wandb.init(project="MLP-single-test", job_type="evaluation")

    best_model = MLPClassifier.load(path="best_model_single.json")
    best_model.fit(X_train, y_train_oh, X_val=X_val, y_val=y_val_oh)

    test_predictions_prob = best_model.predict(X_test)
    test_predictions = np.argmax(test_predictions_prob, axis=1) + 3
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average="macro")
    test_precision = precision_score(y_test, test_predictions, average="macro")
    test_recall = recall_score(y_test, test_predictions, average="macro")

    print(f"Test accuracy of best model: {test_accuracy}")
    print(f"Test F1 score of best model: {test_f1}")
    print(f"Test precision of best model: {test_precision}")
    print(f"Test recall of best model: {test_recall}")

    wandb.log(
        {
            "test_accuracy": test_accuracy,
            "test_f1_score": test_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
        }
    )
    wandb.finish()


# evaluate_best_model_on_test()


def conv_analysis():
    activations = ["relu", "tanh", "sigmoid", "linear"]
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    batch_sizes = [16, 32, 64, 128]

    # Effect of Non-linearity: Activation functions
    plt.figure(figsize=(10, 6))
    for activation in activations:
        model = MLPClassifier.load(
            "best_model_single.json"
        )  # Reload base model for each experiment
        model.activation = activation
        model.fit(X_train, y_train_oh, X_val=X_val, y_val=y_val_oh)
        plt.plot(model.loss_history, label=f"{activation}")
    plt.title("Effect of Activation Functions on Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Effect of Learning Rate
    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        model = MLPClassifier.load(
            "best_model_single.json"
        )  # Reload base model for each experiment
        model.learning_rate = lr
        model.fit(X_train, y_train_oh, X_val=X_val, y_val=y_val_oh)
        plt.plot(model.loss_history, label=f"LR: {lr}")
    plt.title("Effect of Learning Rates on Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Effect of Batch Size
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        model = MLPClassifier.load(
            "best_model_single.json"
        )  # Reload base model for each experiment
        model.batch_size = batch_size
        model.fit(X_train, y_train_oh, X_val=X_val, y_val=y_val_oh)
        plt.plot(model.loss_history, label=f"Batch Size: {batch_size}")
    plt.title("Effect of Batch Size on Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# conv_analysis()


def conv_analysis_2():
    activations = ["relu", "tanh", "sigmoid", "linear"]
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    batch_sizes = [16, 32, 64, 128]

    # Initialize W&B project
    wandb.init(project="MLP-Single-Experiment", name="Activation_LR_Batchsize_Analysis")

    model = MLPClassifier.load("best_model_single.json")

    # Activation function experiments
    for activation in activations:
        model.activation = activation
        with wandb.init(
            project="MLP-Single-Experiment", name=f"Activation-{activation}"
        ):
            model.fit(X_train, y_train_oh, X_val=X_val, y_val=y_val_oh)

    # Learning rate experiments
    for lr in learning_rates:
        model.learning_rate = lr
        with wandb.init(project="MLP-Single-Experiment", name=f"LR-{lr}"):
            model.fit(X_train, y_train_oh, X_val=X_val, y_val=y_val_oh)

    # Batch size experiments
    for batch_size in batch_sizes:
        model.batch_size = batch_size
        with wandb.init(project="MLP-Single-Experiment", name=f"Batch-{batch_size}"):
            model.fit(X_train, y_train_oh, X_val=X_val, y_val=y_val_oh)


# conv_analysis_2()


from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler


def label_encode(df, column):
    labels = df[column].astype("category").cat.codes
    return labels


def preprocess_advert_data(data):
    labels = data["labels"].str.split(" ")
    unique_labels = set()
    for label in labels:
        unique_labels.update(label)

    data.drop(columns=["city"], inplace=True)

    categorical_columns = [
        "gender",
        "education",
        "married",
        # "city",
        "occupation",
        "most bought item",
    ]

    for column in categorical_columns:
        data[column] = label_encode(data, column)

    # data = data.fillna(data.mean())

    # labels_df = one_hot_encode(data['labels'], len(unique_labels))
    mlb = MultiLabelBinarizer()
    labels_df = pd.DataFrame(mlb.fit_transform(labels), columns=mlb.classes_)
    data = pd.concat([data, labels_df], axis=1)
    data.drop(columns=["labels"], inplace=True)

    print(data.head())

    scaler = StandardScaler()
    data.iloc[:, :-8] = pd.DataFrame(scaler.fit_transform(data.iloc[:, :-8]))

    # scaler = MinMaxScaler()
    # data.iloc[:, :-8] = pd.DataFrame(scaler.fit_transform(data.iloc[:, :-8]))

    return data


def advertisement_data():
    df = pd.read_csv("../../data/external/advertisement.csv")
    df = preprocess_advert_data(df)
    return df


# df = advertisement_data()
# print(df.head())
# with open("../../data/interim/3/advertisement_data.csv", "w") as f:
#     df.to_csv(f, index=False)


def evaluate_best_model_on_test_2(X_train, y_train, X_val, y_val, X_test, y_test):
    # best_model = MLPClassifier.load("best_model.pkl")
    # wandb.init(project="MLP-multi-test", job_type="evaluation")

    best_model = MultiLabelMLPClassifier.load(path="best_model_multi.json")
    best_model.wandb_log = False
    best_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # test_metrics = best_model.evaluate(X_test, y_test)
    # print("Test Metrics:", test_metrics)
    # wandb.log(test_metrics)
    # wandb.finish()

    # Get predictions and ground truth
    y_pred_proba = best_model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    y_true = y_test

    metrics_per_class = []
    class_precisions = []
    class_recalls = []
    class_f1_scores = []

    # List of class names
    classes = [
        "beauty",
        "books",
        "clothing",
        "electronics",
        "food",
        "furniture",
        "home",
        "sports",
    ]

    # Calculate TP, TN, FP, FN for each class
    for i in range(y_true.shape[1]):
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        tn = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        metrics_per_class.append((tp, tn, fp, fn))

    # Print class-wise metrics
    print("\nClass-wise Performance Metrics:")
    for i, (tp, tn, fp, fn) in enumerate(metrics_per_class):
        # Calculate accuracy, precision, recall, and F1 score
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0
        )

        # Store per-class metrics for later use in overall calculations
        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1_scores.append(f1)

        # Display metrics for each class
        print(
            f"{classes[i]}: "
            f"Accuracy: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"F1 Score: {f1:.4f}, "
            f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}"
        )

    # Calculate overall (macro-average) precision, recall, and F1 score
    overall_precision = np.mean(class_precisions)
    overall_recall = np.mean(class_recalls)
    overall_f1 = np.mean(class_f1_scores)

    # Display overall metrics
    print("\nOverall Performance Metrics (Macro Average):")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")


def multilabelMLP():
    with open("../../data/interim/3/advertisement_data.csv", "r") as f:
        data = pd.read_csv(f)
    num_label_columns = 8  # Adjust if necessary
    X = data.iloc[:, :-num_label_columns]
    Y = data.iloc[:, -num_label_columns:]

    # Split into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    X_test, X_val, Y_test, Y_val = train_test_split(
        X_test, Y_test, test_size=0.5, random_state=42
    )
    Y_train = Y_train.to_numpy()
    Y_test = Y_test.to_numpy()
    Y_val = Y_val.to_numpy()

    wandb.init(project="MLP-multi-run")
    wandb.config.update(
        {
            "learning_rate": 0.001,
            "epochs": 100,
            "hidden_layers": [64, 32],
            "activation_function": "relu",
            "optimizer": "sgd",
        }
    )

    mlp_classifier = MultiLabelMLPClassifier(
        input_size=X_train.shape[1],
        output_size=Y_train.shape[1],
        hidden_layers=[64, 32],
        activation="sigmoid",
        optimizer="sgd",
        learning_rate=0.01,
        batch_size=32,
        epochs=1000,
        early_stopping=True,
        patience=10,
        wandb_log=True,
    )

    # Train the classifier
    mlp_classifier.fit(X_train, Y_train, X_val=X_val, y_val=Y_val)
    wandb.finish()
    # train_predictions = mlp_classifier.predict(X_train)
    # test_predictions = mlp_classifier.predict(X_test)

    # Compute evaluation metrics for training set
    train_metrics = mlp_classifier.evaluate(X_train, Y_train)
    print("Training Metrics:", train_metrics)

    # Compute evaluation metrics for test set
    test_metrics = mlp_classifier.evaluate(X_test, Y_test)
    print("Test Metrics:", test_metrics)

    hyperparameter_tuning_sweep_multi(X_train, Y_train, X_val, Y_val)

    evaluate_best_model_on_test_2(X_train, Y_train, X_val, Y_val, X_test, Y_test)


# multilabelMLP()


def housing_data():
    df = pd.read_csv("../../data/external/HousingData.csv")
    stats = df.describe().loc[["mean", "std", "min", "max"]]

    for column in stats.columns:
        print(
            f"{column}:\nMean: {stats[column]['mean']}\nStd: {stats[column]['std']}\nMin: {stats[column]['min']}\nMax: {stats[column]['max']}\n"
        )

    plt.figure(figsize=(15, 12))

    attributes = df.columns
    for i, col in enumerate(attributes):
        plt.subplot(4, 4, i + 1)
        sns.histplot(df[col], kde=True, color="blue", edgecolor="black")
        plt.title(col)

    plt.tight_layout()
    plt.savefig("figures/housing_data.png")

    df.fillna(df.mean(), inplace=True)
    features = df.drop(columns=["MEDV"])

    standard_scaler = StandardScaler()
    # standardized_data = standard_scaler.fit_transform(features)
    # standardized_df = pd.DataFrame(standardized_data, columns=features.columns)
    # standardized_df["MEDV"] = df["MEDV"]
    standardized_df = standard_scaler.fit_transform(df)
    standardized_df = pd.DataFrame(standardized_df, columns=df.columns)

    print(standardized_df.head())

    plt.figure(figsize=(12, 6))
    corr = standardized_df.corr()
    sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
    plt.title("Correlation Matrix")
    plt.savefig("figures/correlation_matrix_housing.png")

    return standardized_df


# housing_dataset = housing_data()
# with open("../../data/interim/3/HousingData.csv", "w") as f:
#     housing_dataset.to_csv(f, index=False)
with open("../../data/interim/3/HousingData.csv", "r") as f:
    housing_dataset = pd.read_csv(f)

X_train, X_test, y_train, y_test = train_test_split(
    housing_dataset.drop(columns=["MEDV"]).values,
    housing_dataset["MEDV"].values,
    test_size=0.2,
    random_state=42,
)
X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)
y_train = y_train.reshape(-1, 1)

from sklearn.metrics import r2_score, mean_absolute_error


def mlp_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    mlp_model = MLPClassifier(
        input_size=X_train.shape[1],
        output_size=1,
        hidden_layers=[64, 32],
        activation="sigmoid",
        optimizer="mini_batch",
        learning_rate=0.01,
        batch_size=32,
        epochs=100,
        early_stopping=True,
        patience=10,
        wandb_log=True,
        task_type="regression",
    )
    # hidden_layers = [20]
    # input_size = X_train.shape[1]
    # output_size = 1  # Adjust this based on your regression task

    # mlp_model = MLPRegression(
    #     input_size,
    #     hidden_layers,
    #     output_size,
    #     epochs=100,
    #     learning_rate=0.01,
    #     activation="sigmoid",
    #     optimizer="batch",
    #     batch_size=59,
    # )
    # y_train = y_train.reshape(-1, 1)

    wandb.init(project="MLP-regression-first-r")
    wandb.config.update(
        {
            "learning_rate": 0.001,
            "epochs": 100,
            "hidden_layers": [20],
            "activation_function": "sigmoid",
            "optimizer": "mini_batch",
        }
    )

    mlp_model.fit(X_train, y_train, X_val, y_val)
    predictions = mlp_model.predict(X_test)

    mse = np.mean((y_test - predictions) ** 2)

    rmse = np.sqrt(mse)

    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    # r_squared = 1 - (ss_res / ss_tot)
    r_2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-Squared: {r_2}")
    wandb.log({"MSE": mse, "RMSE": rmse, "R2": r_2})

    wandb.finish()


# mlp_regression(X_train, y_train, X_val, y_val, X_test, y_test)


def hyperparameter_tuning_regression(X_train, y_train, X_val, y_val):
    sweep_config = {
        "method": "grid",
        "metric": {"name": "MSE", "goal": "minimize"},
        "parameters": {
            "layers": {
                "values": [[16], [64, 32], [16, 32, 16], [32, 16], [64, 32, 16, 8]]
            },
            "learning_rate": {"values": [0.01, 0.001]},
            "activation": {"values": ["sigmoid", "tanh", "relu"]},
            "optimizer": {"values": ["sgd", "batch", "mini_batch"]},
        },
    }

    sweep_id = wandb.sweep(
        sweep_config, project="MLP-HyperparameterTuningRegression-fi"
    )

    def train_model():
        wandb.init()
        config = wandb.config

        print(f"Running with config: {config}")
        wandb.run.name = f"{config['activation']}-{config['optimizer']}-{config['layers']}-{config['learning_rate']}"

        model = MLPClassifier(
            input_size=X_train.shape[1],
            output_size=1,
            hidden_layers=config["layers"],
            activation=config["activation"],
            optimizer=config["optimizer"],
            learning_rate=config["learning_rate"],
            epochs=100,
            early_stopping=True,
            patience=10,
            wandb_log=True,
            task_type="regression",
        )

        model.fit(X_train, y_train, X_val, y_val)

        predictions = model.predict(X_val)

        mse = np.mean((y_val - predictions) ** 2)

        rmse = np.sqrt(mse)

        ss_res = np.sum((y_val - predictions) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        # r_squared = 1 - (ss_res / ss_tot)
        mae = np.mean(np.abs(y_val - predictions))
        r_2 = r2_score(y_val, predictions)
        wandb.log({"MSE": mse, "RMSE": rmse, "R2": r_2, "MAE": mae})

        wandb.finish()

    wandb.agent(sweep_id, function=train_model)


hyperparameter_tuning_regression(X_train, y_train, X_val, y_val)


def evaluate_regression_on_test():
    # wandb.init(project="MLP-regression-test", job_type="evaluation")

    best_model = MLPClassifier.load(path="best_model_regression.json")
    best_model.wandb_log = False
    best_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    test_predictions = best_model.predict(X_test)

    mse = np.mean((y_test - test_predictions) ** 2)

    rmse = np.sqrt(mse)

    ss_res = np.sum((y_test - test_predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    # r_squared = 1 - (ss_res / ss_tot)
    r2 = r2_score(y_test, test_predictions)
    # MAE
    mae = np.mean(np.abs(y_test - test_predictions))
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-Squared: {r2}")
    print(f"Mean Absolute Error: {mae}")
    individual_losses = np.array(
        [(y_test[i] - test_predictions[i]) ** 2 for i in range(len(y_test))]
    )

    # Create a DataFrame with features, true values, predictions, and losses
    feature_names = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
    ]

    results_df = pd.DataFrame(X_test, columns=feature_names)
    results_df["true_value"] = y_test
    results_df["prediction"] = test_predictions
    results_df["mse_loss"] = individual_losses

    # Sort by loss
    sorted_results = results_df.sort_values("mse_loss", ascending=False)
    print("Top 10 Data Points with Highest MSE Loss:")
    print(sorted_results.head(10))
    sorted_results.head(10).to_csv("figures/top_10.csv")

    # Print the bottom 10 data points with lowest MSE loss
    print("\nBottom 10 Data Points with Lowest MSE Loss:")
    print(sorted_results.tail(10))
    sorted_results.tail(10).to_csv("figures/bottom_10.csv")

    results_df["mse_loss"] = (results_df["true_value"] - results_df["prediction"]) ** 2

    plt.figure(figsize=(10, 6))
    plt.scatter(
        np.arange(len(results_df)),
        results_df["mse_loss"],
        color="blue",
        alpha=0.6,
        label="MSE Loss",
    )
    plt.title("Mean Squared Error vs Data Points")
    plt.xlabel("Data Point Index")
    plt.ylabel("Mean Squared Error Loss")
    plt.axhline(
        y=np.mean(results_df["mse_loss"]),
        color="red",
        linestyle="--",
        label="Mean MSE Loss",
    )
    plt.legend()
    plt.grid()
    plt.savefig("figures/mse_vs_datapoints.png")

    # wandb.log({"MSE": mse, "RMSE": rmse, "R2": r2, "MAE": mae})
    # wandb.finish()


# Binary Classification
def diabetes_data():
    df = pd.read_csv("../../data/external/diabetes.csv")

    df.fillna(df.mean(), inplace=True)
    features = df.drop(columns=["Outcome"])

    standard_scaler = StandardScaler()
    standardized_data = standard_scaler.fit_transform(features)
    standardized_df = pd.DataFrame(standardized_data, columns=features.columns)
    standardized_df["Outcome"] = df["Outcome"]

    print(standardized_df.head())

    return standardized_df


# diabetes_dataset = diabetes_data()
# with open("../../data/interim/3/diabetes.csv", "w") as f:
#     diabetes_dataset.to_csv(f, index=False)
with open("../../data/interim/3/diabetes.csv", "r") as f:
    diabetes_dataset = pd.read_csv(f)


X_train, X_test, y_train, y_test = train_test_split(
    diabetes_dataset.drop(columns=["Outcome"]).values,
    diabetes_dataset["Outcome"].values,
    test_size=0.2,
    random_state=42,
)
X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)


def bce_vs_mse():
    # Model 1: BCE Loss
    y_train_oh = one_hot_encode(y_train, 2, a=0)
    y_val_oh = one_hot_encode(y_val, 2, a=0)
    y_test_oh = one_hot_encode(y_test, 2, a=0)

    model_bce = MLPClassifier(
        input_size=X_train.shape[1],
        output_size=2,
        hidden_layers=[],
        activation="relu",
        optimizer="mini_batch",
        learning_rate=0.01,
        epochs=100,
        loss_function="bce",
        wandb_log=False,
    )
    model_bce.fit(X_train, y_train_oh, X_val, y_val_oh)

    # Model 2: MSE Loss
    model_mse = MLPClassifier(
        input_size=X_train.shape[1],
        output_size=2,
        hidden_layers=[],
        activation="relu",
        optimizer="mini_batch",
        learning_rate=0.01,
        epochs=100,
        loss_function="mse",
        wandb_log=False,
    )
    model_mse.fit(X_train, y_train_oh, X_val, y_val_oh)
    plt.figure(figsize=(10, 5))
    plt.plot(model_bce.loss_history, label="BCE Loss")
    plt.title("Loss vs Epochs (BCE)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig("figures/bce_loss.png")

    # Plot MSE loss
    plt.figure(figsize=(10, 5))
    plt.plot(model_mse.loss_history, label="MSE Loss")
    plt.title("Loss vs Epochs (MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig("figures/mse_loss.png")

    plt.figure(figsize=(10, 5))
    plt.plot(model_bce.loss_history, label="BCE Loss", color="blue")
    plt.plot(model_mse.loss_history, label="MSE Loss", color="green")
    plt.title("Loss vs Epochs (BCE vs MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("figures/bce_vs_mse_loss.png")

    # using task_type="regression" in the MLPClassifier constructor
    model_bce = MLPClassifier(
        input_size=X_train.shape[1],
        output_size=2,
        hidden_layers=[],
        activation="relu",
        optimizer="mini_batch",
        learning_rate=0.01,
        epochs=100,
        loss_function="bce",
        wandb_log=False,
        task_type="regression",
    )
    y_train2 = y_train.reshape(-1, 1)
    y_val2 = y_val.reshape(-1, 1)
    model_bce.fit(X_train, y_train2, X_val, y_val2)

    # Model 2: MSE Loss
    model_mse = MLPClassifier(
        input_size=X_train.shape[1],
        output_size=2,
        hidden_layers=[],
        activation="relu",
        optimizer="mini_batch",
        learning_rate=0.01,
        epochs=100,
        loss_function="mse",
        wandb_log=False,
        task_type="regression",
    )
    model_mse.fit(X_train, y_train2, X_val, y_val2)
    plt.figure(figsize=(10, 5))
    plt.plot(model_bce.loss_history, label="BCE Loss")
    plt.title("Loss vs Epochs (BCE)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig("figures/bce_loss_r.png")

    # Plot MSE loss
    plt.figure(figsize=(10, 5))
    plt.plot(model_mse.loss_history, label="MSE Loss")
    plt.title("Loss vs Epochs (MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig("figures/mse_loss_r.png")

    plt.figure(figsize=(10, 5))
    plt.plot(model_bce.loss_history, label="BCE Loss", color="blue")
    plt.plot(model_mse.loss_history, label="MSE Loss", color="green")
    plt.title("Loss vs Epochs (BCE vs MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("figures/bce_vs_mse_loss_r.png")


# bce_vs_mse()


def normalize_data(df, features):
    df_normalized = df.copy()
    for feature in features:
        mean_value = df[feature].mean()
        std_dev = df[feature].std()
        df_normalized[feature] = (df[feature] - mean_value) / std_dev
    return df_normalized


from performance_measures.knn_metrics import ClassificationMetrics


def spotify_dataset():
    df = pd.read_csv("../../data/external/spotify.csv")

    num_features = [
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "key",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]
    # # Encode categorical variables
    # df["track_genre"], genre_mapping = pd.factorize(df["track_genre"])
    # df = normalize_data(df, num_features)
    # df = df.drop(
    #     columns=["track_id", "artists", "album_name", "track_name", "Unnamed: 0"]
    # )
    # df = df.sample(frac=1, random_state=42)
    # df = df.reset_index(drop=True)
    # df.to_csv("../../data/interim/3/spotify.csv", index=False)
    df = pd.read_csv("../../data/interim/1/spotify.csv")
    X = df.drop(columns=["track_genre"]).values

    y = df["track_genre"].values

    input_size = X.shape[1]  # Number of features in the input
    latent_size = 10  # Adjust based on the optimal PCA dimensions from Assignment 2

    # autoencoder = AutoEncoder(input_size=input_size, latent_dim=latent_size)

    # # Train the autoencoder
    # autoencoder.fit(X, None)
    # # Get latent representation and reconstruct data
    # latent_representation = autoencoder.get_latent(X)
    # reconstructed_data = autoencoder.reconstruct(X)
    # latent_df = pd.DataFrame(latent_representation)

    # print("Latent representation shape:", latent_representation.shape)
    # print("Reconstructed data shape:", reconstructed_data.shape)

    # with open("../../data/interim/3/spotify_latent_representation.csv", "w") as f:
    #     np.savetxt(f, latent_representation, delimiter=",")

    # reconstruction_loss = np.mean((X - reconstructed_data) ** 2)
    # print("Reconstruction loss:", reconstruction_loss)
    with open("../../data/interim/3/spotify_latent_representation.csv", "r") as f:
        latent_df = pd.read_csv(f, header=None)

    print(latent_df.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        latent_df, y, test_size=0.2, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    y_train = y_train.reshape(-1, 1)

    X_train = X_train.values
    X_test = X_test.values
    X_val = X_val.values
    # model = KNN(k=24, distance_metric="manhattan")
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_val)["predictions"]

    # metrics = ClassificationMetrics()
    # accuracy = metrics.accuracy(y_val, y_pred)
    # print("Validation Accuracy:", accuracy)

    # y_pred = model.predict(X_test)["predictions"]

    # metrics = ClassificationMetrics()
    # accuracy = metrics.accuracy(y_test, y_pred)
    # print("Test Accuracy:", accuracy)

    n_classes = len(np.unique(y))

    def one_hot_encode(y, num_classes, a=3):
        return np.eye(num_classes)[y - a]  # Shift by 3 since quality labels start at 3

    # One-hot encode the labels for training and testing
    y_train_oh = one_hot_encode(y_train, n_classes, a=0)
    y_test_oh = one_hot_encode(y_test, n_classes, a=0)
    y_val_oh = one_hot_encode(y_val, n_classes, a=0)

    # mlp = MLPClassifier(
    #     input_size=X_train.shape[1],
    #     output_size=len(np.unique(y_train)),
    #     hidden_layers=[64, 32],
    #     activation="sigmoid",
    #     optimizer="sgd",
    #     learning_rate=0.01,
    #     batch_size=128,
    #     epochs=100,
    #     early_stopping=True,
    #     patience=5,
    #     wandb_log=False,
    # )
    # mlp.fit(X_train, y_train_oh, X_val=X_val, y_val=y_val_oh)

    # # Predict on the test set (probabilities)
    # y_pred_proba = mlp.predict(X_test)

    # # Convert predicted probabilities back to class labels
    # y_pred = np.argmax(y_pred_proba, axis=1)

    # accuracy = accuracy_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred, average="weighted")
    # precision = precision_score(y_test, y_pred, average="weighted")
    # recall = recall_score(y_test, y_pred, average="weighted")

    # print(f"Test Accuracy: {accuracy:.4f}")
    # print(f"Test F1 Score: {f1:.4f}")
    # print(f"Test Precision: {precision:.4f}")
    # print(f"Test Recall: {recall:.4f}")

    # # Detailed classification report
    # print(classification_report(y_test, y_pred))
    # plt.figure(figsize=(10, 6))
    # plt.plot(mlp.loss_history)
    # plt.title("Training Loss History")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.savefig("figures/spotify_loss.png")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )

    mlp = MLPClassifier(
        input_size=X_train.shape[1],
        output_size=len(np.unique(y_train)),
        hidden_layers=[64, 32, 16],
        activation="sigmoid",
        optimizer="sgd",
        learning_rate=0.01,
        batch_size=128,
        epochs=100,
        early_stopping=True,
        patience=5,
        wandb_log=False,
    )
    mlp.fit(X_train, y_train_oh, X_val=X_val, y_val=y_val_oh)

    # Predict on the test set (probabilities)
    y_pred_proba = mlp.predict(X_test)

    # Convert predicted probabilities back to class labels
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")

    # Detailed classification report
    print(classification_report(y_test, y_pred))
    plt.figure(figsize=(10, 6))
    plt.plot(mlp.loss_history)
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("figures/spotify_loss_2.png")


# spotify_dataset()
