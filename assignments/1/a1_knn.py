import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import time
import itertools
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.abspath("../../"))

from models.knn import knn
from performance_measures import knn_metrics


def perform_eda(df):

    # List of numerical features
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

    # Plot distributions of numerical features
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(num_features):
        plt.subplot(3, 4, i + 1)
        sns.histplot(df[feature], kde=True)
        plt.title(feature)
    plt.tight_layout()
    plt.savefig("figures/histograms_2.png")
    plt.close()

    # Boxplots for numerical features
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(num_features):
        plt.subplot(3, 4, i + 1)
        sns.boxplot(x=df[feature])
        plt.title(f"Boxplot of {feature}")
        plt.xlabel(feature)
    plt.tight_layout()
    plt.savefig("figures/boxplots_2.png")
    plt.close()

    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(num_features):
        plt.subplot(3, 4, i + 1)
        sns.violinplot(y=df[feature])
        plt.title(f"Violin Plot of {feature}")
    plt.tight_layout()
    plt.savefig("figures/violin_plots_2.png")
    plt.close()

    # Distribution of 'explicit'
    plt.figure(figsize=(6, 4))
    sns.countplot(x="explicit", data=df)
    plt.title("Distribution of Explicit Content")
    plt.savefig("figures/explicit_distribution_2.png")
    plt.close()

    # Correlation matrix
    correlation_matrix = df[num_features + ["track_genre"]].copy()
    correlation_matrix["track_genre"] = pd.factorize(df["track_genre"])[0]
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.savefig("figures/correlation_matrix_2.png")
    plt.close()

    unique_genres = df["track_genre"].unique()
    num_genres = len(unique_genres)
    genres_per_plot = num_genres // 4 + 1
    genre_subsets = [
        unique_genres[i : i + genres_per_plot]
        for i in range(0, num_genres, genres_per_plot)
    ]

    # Plot energy vs. track_genre in 2x2 grid
    plt.figure(figsize=(15, 12))
    for idx, subset in enumerate(genre_subsets):
        plt.subplot(2, 2, idx + 1)
        sns.boxplot(
            x="track_genre", y="energy", data=df[df["track_genre"].isin(subset)]
        )
        plt.xticks(rotation=90)
        plt.title(f"Energy Distribution (Genres {idx + 1})")
        plt.tight_layout()
    plt.savefig("figures/energy_vs_genre_grid_2.png")
    plt.close()


def normalize_data(df, features):
    df_normalized = df.copy()
    for feature in features:
        mean_value = df[feature].mean()
        std_dev = df[feature].std()
        df_normalized[feature] = (df[feature] - mean_value) / std_dev
    return df_normalized


def evaluate_feature_combinations(X_train, y_train, X_val, y_val):
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

    best_accuracy = 0
    best_features = []

    # Test all possible combinations of features
    for i in range(1, len(num_features) + 1):
        print(i)
        for subset in itertools.combinations(num_features, i):
            X_train_subset = X_train[:, [num_features.index(f) for f in subset]]
            X_val_subset = X_val[:, [num_features.index(f) for f in subset]]

            model = knn.KNN(k=24, distance_metric="manhattan")
            model.fit(X_train_subset, y_train)
            y_pred = model.predict(X_val_subset)["predictions"]

            metrics = knn_metrics.ClassificationMetrics()
            accuracy = metrics.accuracy(y_val, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features = subset

    return best_features, best_accuracy


def remove_subset(X_train, y_train, X_val, y_val, remove_features):
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
    best_accuracy = 0
    current_features = [f for f in num_features if f not in remove_features]
    X_train_subset = X_train[:, [num_features.index(f) for f in current_features]]
    X_val_subset = X_val[:, [num_features.index(f) for f in current_features]]

    model = knn.KNN(k=24, distance_metric="manhattan")
    model.fit(X_train_subset, y_train)
    y_pred = model.predict(X_val_subset)["predictions"]

    # Calculate accuracy
    metrics = knn_metrics.ClassificationMetrics()
    accuracy = metrics.accuracy(y_val, y_pred)

    print(
        "Accuracy with features: ",
        current_features,
        "after removing: ",
        remove_features,
    )
    print(accuracy)


def find_best_feature_subset(X_train, y_train, X_val, y_val):
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
    best_accuracy = 0
    best_features = num_features.copy()

    # Start with all features and drop one at a time
    for feature in num_features:
        print("Feature removed: ", feature)
        # Create a subset excluding the current feature
        current_features = [f for f in num_features if f != feature]
        X_train_subset = X_train[:, [num_features.index(f) for f in current_features]]
        X_val_subset = X_val[:, [num_features.index(f) for f in current_features]]

        model = knn.KNN(k=24, distance_metric="manhattan")
        model.fit(X_train_subset, y_train)
        y_pred = model.predict(X_val_subset)["predictions"]

        # Calculate accuracy
        metrics = knn_metrics.ClassificationMetrics()
        accuracy = metrics.accuracy(y_val, y_pred)

        # Update the best accuracy and feature set if current accuracy is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_features = current_features
        print(f"Accuracy: {accuracy}")

    return best_features, best_accuracy


def preprocess_data(df, shuffle=True, name="../../data/interim/1/spotify.csv"):

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
    # Encode categorical variables
    df["track_genre"], genre_mapping = pd.factorize(df["track_genre"])
    df = normalize_data(df, num_features)
    df = df.drop(
        columns=["track_id", "artists", "album_name", "track_name", "Unnamed: 0"]
    )
    if shuffle == True:
        df = df.sample(frac=1, random_state=42)
    df = df.reset_index(drop=True)
    if name:
        df.to_csv(name, index=False)

    return df


def split_data(X, y, train_size=0.8, val_size=0.1):

    # Calculate split indices
    train_end = int(train_size * len(X))
    val_end = train_end + int(val_size * len(X))

    # Split the data
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_knn(X_train, y_train, X_val, y_val, k_values, distance_metrics):
    results = []

    for metric in distance_metrics:
        accuracies = []
        for k in k_values:
            model = knn.KNN(k=k, distance_metric=metric)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)["predictions"]

            # Calculate metrics
            metrics = knn_metrics.ClassificationMetrics()
            accuracy = metrics.accuracy(y_val, y_pred)
            accuracies.append(accuracy)
            results.append((k, metric, accuracy))
            precision = metrics.precision(y_val, y_pred)
            recall = metrics.recall(y_val, y_pred)
            f1_mac = metrics.f1_score(y_val, y_pred, average="macro")
            f1_mic = metrics.f1_score(y_val, y_pred, average="micro")

            print(f"K: {k}, Metric: {metric}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score Macro: {f1_mac}")
            print(f"F1 Score Micro: {f1_mic}")

        if metric == "manhattan":
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, accuracies, marker="o")
            plt.title(f"k vs Accuracy Manhattan")
            plt.xlabel("k")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.savefig("figures/k_vs_accuracy.png")
            plt.close()

    # Convert results to DataFrame and sort
    results_df = pd.DataFrame(results, columns=["k", "distance_metric", "accuracy"])
    top_results = results_df.sort_values(by="accuracy", ascending=False).head(10)

    return top_results


def compare_knn_inference_times(X_train, y_train, X_test):

    print("Comparing inference times")

    # Best KNN model (k=24, distance=manhattan), optimized
    best_knn = knn.KNN(k=24, distance_metric="manhattan", optimization_type="optimized")
    best_knn.fit(X_train, y_train)
    best_results = best_knn.predict(X_test)
    best_avg_inference_time = best_results["inference_time"] / len(X_test)
    print("Best model: ", best_avg_inference_time)

    # Most optimized KNN model (default settings, optimized code)
    optimized_knn = knn.KNN(
        k=3, distance_metric="manhattan", optimization_type="optimized"
    )
    optimized_knn.fit(X_train, y_train)
    optimized_results = optimized_knn.predict(X_test)
    optimized_avg_inference_time = optimized_results["inference_time"] / len(X_test)
    print("Optimised model: ", optimized_avg_inference_time)

    # Default sklearn KNN model
    sklearn_knn = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
    sklearn_knn.fit(X_train, y_train)
    sklearn_start_time = time.time()
    sklearn_preds = sklearn_knn.predict(X_test)
    sklearn_inference_time = time.time() - sklearn_start_time
    sklearn_avg_inference_time = sklearn_inference_time / len(X_test)
    print("sklearn model: ", sklearn_avg_inference_time)

    # Naive KNN model
    naive_knn = knn.KNN(k=3, distance_metric="manhattan", optimization_type="naive")
    naive_knn.fit(X_train, y_train)
    naive_results = naive_knn.predict(X_test)
    naive_avg_inference_time = naive_results["inference_time"] / len(X_test)
    print("Naive model: ", naive_avg_inference_time)

    models = ["Naive KNN", "Best KNN", "Optimized KNN", "Sklearn KNN"]
    avg_inference_times = [
        naive_avg_inference_time,
        best_avg_inference_time,
        optimized_avg_inference_time,
        sklearn_avg_inference_time,
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(models, avg_inference_times, color=["blue", "green", "red", "purple"])
    plt.xlabel("KNN Model")
    plt.ylabel("Average Inference Time per Sample (seconds)")
    plt.title("Comparison of Average Inference Times for Different KNN Models")
    plt.show()


def preprocess_spotify_2(combine=True):

    data_train = pd.read_csv("../../data/external/spotify-2/train.csv")
    data_val = pd.read_csv("../../data/external/spotify-2/validate.csv")
    data_test = pd.read_csv("../../data/external/spotify-2/test.csv")
    if combine:

        combined_data = pd.concat([data_train, data_test, data_val], ignore_index=True)

        for col in combined_data.select_dtypes(include=["number"]):
            combined_data[col] = (
                combined_data[col] - combined_data[col].mean()
            ) / combined_data[col].std()

        # perform_eda(combined_data)
        df = preprocess_data(combined_data, shuffle=False, name=None)

        data_train = df.iloc[: len(data_train)]
        data_test = df.iloc[len(data_train) : len(data_train) + len(data_test)]
        data_val = df.iloc[len(data_train) + len(data_test) :]

    else:
        data_train = preprocess_data(data_train, shuffle=True, name=None)
        data_val = preprocess_data(data_val, shuffle=True, name=None)
        data_test = preprocess_data(data_test, shuffle=True, name=None)

    data_train.to_csv("../../data/interim/1/spotify2/train1.csv", index=False)
    data_val.to_csv("../../data/interim/1/spotify2/validate1.csv", index=False)
    data_test.to_csv("../../data/interim/1/spotify2/test1.csv", index=False)


def run_on_spotify_2(k, metric):

    data_train = pd.read_csv("../../data/interim/1/spotify2/train1.csv")
    data_val = pd.read_csv("../../data/interim/1/spotify2/validate1.csv")
    data_test = pd.read_csv("../../data/interim/1/spotify2/test1.csv")

    X_train = data_train.drop(columns=["track_genre"]).values

    y_train = data_train["track_genre"].values

    X_val = data_val.drop(columns=["track_genre"]).values

    y_val = data_val["track_genre"].values
    model = knn.KNN(k=k, distance_metric=metric)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)["predictions"]

    # Calculate metrics
    metrics = knn_metrics.ClassificationMetrics()
    accuracy = metrics.accuracy(y_val, y_pred)
    precision = metrics.precision(y_val, y_pred)
    recall = metrics.recall(y_val, y_pred)
    f1_mac = metrics.f1_score(y_val, y_pred, average="macro")
    f1_mic = metrics.f1_score(y_val, y_pred, average="micro")

    print(f"K: {k}, Metric: {metric}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score Macro: {f1_mac}")
    print(f"F1 Score Micro: {f1_mic}")

    print("Accuracy: ", accuracy)


def main():
    preprocess_spotify_2(combine=False)
    run_on_spotify_2(24, "manhattan")
    # Load dataset
    df = pd.read_csv("../../data/external/spotify.csv")
    # df = df.replace({True: 1, False: 0})

    perform_eda(df)
    # ======================================================

    # Preprocess data
    # X, y, genre_mapping = preprocess_data(df)
    # X = df.drop(columns=["track_genre"])

    # y = df["track_genre"]
    df = pd.read_csv("../../data/interim/1/spotify.csv")
    X = df.drop(columns=["track_genre"]).values

    y = df["track_genre"].values

    # # Split data into training and validation sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    k_values = range(1, 25)
    distance_metrics = ["euclidean", "manhattan", "cosine"]

    top_results = evaluate_knn(
        X_train, y_train, X_val, y_val, k_values, distance_metrics
    )
    print("Top 10 Pairs:")
    print(top_results)

    best_features, best_accuracy = find_best_feature_subset(
        X_train, y_train, X_val, y_val
    )
    print(f"Best feature combination: {best_features}")
    print(f"Best accuracy: {best_accuracy}")
    remove_subset(
        X_train,
        y_train,
        X_val,
        y_val,
        remove_features=["loudness", "key"],
    )

    compare_knn_inference_times(X_train, y_train, X_test)


if __name__ == "__main__":
    main()
