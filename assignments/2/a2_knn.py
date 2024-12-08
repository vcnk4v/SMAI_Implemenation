import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.abspath("../../"))


from models.knn import knn
from performance_measures import knn_metrics


def split_data(X, y, train_size=0.8, val_size=0.1):

    # Calculate split indices
    train_end = int(train_size * len(X))
    val_end = train_end + int(val_size * len(X))

    # Split the data
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def measure_inference_time(X_spotify, reduced_data, y):
    X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = split_data(X_spotify, y)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(reduced_data, y)

    knn_red = knn.KNN(k=24, distance_metric="manhattan")
    knn1 = knn.KNN(k=24, distance_metric="manhattan")
    # Measure inference time on original dataset
    knn_red.fit(X_train, y_train)
    knn1.fit(X_train1, y_train1)

    start_time = time.time()
    y_pred = knn_red.predict(X_test)
    inference_time = time.time() - start_time

    # Measure inference time on reduced dataset
    start_time = time.time()
    y_pred_reduced = knn1.predict(X_test1)
    inference_time1 = time.time() - start_time

    plt.figure(figsize=(10, 6))
    plt.bar(
        ["Reduced Dataset", "Original Dataset"],
        [inference_time, inference_time],
        color=["blue", "orange"],
    )
    plt.title("Inference Time for KNN Model")
    plt.ylabel("Inference Time (seconds)")
    plt.show()

    print(f"Inference time on original dataset: {inference_time:.4f} seconds")
    print(f"Inference time on reduced dataset: {inference_time1:.4f} seconds")


def perform_knn_analysis(X_spotify, reduced_data, y):
    X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = split_data(X_spotify, y)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(reduced_data, y)

    model = knn.KNN(k=24, distance_metric="manhattan")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)["predictions"]

    # Calculate metrics
    metrics = knn_metrics.ClassificationMetrics()
    accuracy = metrics.accuracy(y_test, y_pred)
    precision = metrics.precision(y_test, y_pred)
    recall = metrics.recall(y_test, y_pred)
    f1_mac = metrics.f1_score(y_test, y_pred, average="macro")
    f1_mic = metrics.f1_score(y_test, y_pred, average="micro")

    print(f"K: 24, Metric: Manhattan")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score Macro: {f1_mac}")
    print(f"F1 Score Micro: {f1_mic}")

    print("Before PCA:")

    model = knn.KNN(k=24, distance_metric="manhattan")
    model.fit(X_train1, y_train1)
    y_pred = model.predict(X_test1)["predictions"]

    # Calculate metrics
    metrics = knn_metrics.ClassificationMetrics()
    accuracy = metrics.accuracy(y_test1, y_pred)
    precision = metrics.precision(y_test1, y_pred)
    recall = metrics.recall(y_test1, y_pred)
    f1_mac = metrics.f1_score(y_test1, y_pred, average="macro")
    f1_mic = metrics.f1_score(y_test1, y_pred, average="micro")

    print(f"K: 24, Metric: Manhattan")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score Macro: {f1_mac}")
    print(f"F1 Score Micro: {f1_mic}")
