import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath("../../"))

from models.linear_regression import linear_regression
from performance_measures import lin_reg_metrics


def evaluate_model(
    X_train,
    y_train,
    X_test,
    y_test,
    degree,
    learning_rate=0.01,
    epochs=1000,
    regularisation_param=0,
    regularisation_type=None,
    seed=42,
):

    model = linear_regression.LinearRegression(
        degree=degree,
        learning_rate=learning_rate,
        tolerance=1e-6,
        max_iter=epochs,
        regularisation_param=regularisation_param,
        regularisation_type=regularisation_type,
        seed=seed,
    )
    convergence = model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = lin_reg_metrics.RegressionMetrics(y_train, y_train_pred)
    test_metrics = lin_reg_metrics.RegressionMetrics(y_test, y_test_pred)

    train_mse = train_metrics.mse()
    test_mse = test_metrics.mse()
    train_variance = train_metrics.variance()
    test_variance = test_metrics.variance()
    train_std = train_metrics.std_deviation()
    test_std = test_metrics.std_deviation()

    if regularisation_type:
        plt.scatter(X_train, y_train, color="blue", label="Original Data")
        x_range = np.linspace(X_train.min(), X_train.max(), 100)
        y_line = model.predict(x_range)
        plt.plot(x_range, y_line, color="red", label="Fitted Line")
        plt.savefig(f"figures/{regularisation_type}_{degree}.png")
        plt.close()
    return (
        train_mse,
        test_mse,
        train_variance,
        test_variance,
        train_std,
        test_std,
        convergence,
        model,
    )


def learning_rates_experiment(rates, dataset_path, degree):
    (X_train, y_train), (_, _), (X_test, y_test) = split_into_train_test_val(
        dataset_path
    )
    for rate in rates:
        (
            train_mse,
            test_mse,
            train_variance,
            test_variance,
            train_std,
            test_std,
            convergence,
            _,
        ) = evaluate_model(
            X_train, y_train, X_test, y_test, degree=degree, learning_rate=rate
        )

        print(f"Learning rate {rate}:")
        print(f"Degree: {degree}")
        print(f"  Converged after iterations: {convergence}")
        print(f"  Train MSE: {train_mse}")
        print(f"  Test MSE: {test_mse}")
        print(f"  Train Variance: {train_variance}")
        print(f"  Test Variance: {test_variance}")
        print(f"  Train Std Deviation: {train_std}")
        print(f"  Test Std Deviation: {test_std}")
        print()


def seeds_experiment(seeds, dataset_path):
    (X_train, y_train), (_, _), (X_test, y_test) = split_into_train_test_val(
        dataset_path
    )
    for seed in seeds:
        (
            train_mse,
            test_mse,
            train_variance,
            test_variance,
            train_std,
            test_std,
            convergence,
            _,
        ) = evaluate_model(X_train, y_train, X_test, y_test, degree=4, seed=seed)

        print(f"Seed {seed}:")
        print("Degree: 4")
        print(f"  Converged after iterations: {convergence}")
        print(f"  Train MSE: {train_mse}")
        print(f"  Test MSE: {test_mse}")
        print(f"  Train Variance: {train_variance}")
        print(f"  Test Variance: {test_variance}")
        print(f"  Train Std Deviation: {train_std}")
        print(f"  Test Std Deviation: {test_std}")
        print()


def split_into_train_test_val(file_path, train_size=0.8, val_size=0.1, test_size=0.1):
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    np.random.seed(42)
    np.random.shuffle(data)
    X = data[:, 0]
    y = data[:, 1]

    total_samples = len(X)
    train_end = int(train_size * total_samples)
    val_end = train_end + int(val_size * total_samples)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def plot_dataset_splits(X_train, y_train, X_val, y_val, X_test, y_test, name):
    plt.figure(figsize=(10, 6))

    # Plot Training Data
    plt.scatter(X_train, y_train, color="blue", label="Training Set", alpha=0.6)

    # Plot Validation Data
    plt.scatter(X_val, y_val, color="green", label="Validation Set", alpha=0.6)

    # Plot Test Data
    plt.scatter(X_test, y_test, color="red", label="Test Set", alpha=0.6)

    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Dataset Splits Visualization")
    plt.legend()
    plt.savefig(f"figures/dataset_split_{name}")


def run_on_dataset(dataset_path, regularisation_param=0, regularisation_type=None):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_into_train_test_val(
        dataset_path
    )

    plot_dataset_splits(
        X_train, y_train, X_val, y_val, X_test, y_test, regularisation_type
    )

    if regularisation_type:
        degrees = [i for i in range(1, 21)]

    else:
        degrees = [i for i in range(1, 6)]

    best_degree = None
    best_test_mse = float("inf")
    best_model = None
    metrics_list = []

    for degree in degrees:
        (
            train_mse,
            test_mse,
            train_variance,
            test_variance,
            train_std,
            test_std,
            _,
            model,
        ) = evaluate_model(
            X_train,
            y_train,
            X_test,
            y_test,
            degree,
            regularisation_param=regularisation_param,
            regularisation_type=regularisation_type,
        )

        print(f"Degree {degree}:")
        print(f"  Train MSE: {train_mse}")
        print(f"  Test MSE: {test_mse}")
        print(f"  Train Variance: {train_variance}")
        print(f"  Test Variance: {test_variance}")
        print(f"  Train Std Deviation: {train_std}")
        print(f"  Test Std Deviation: {test_std}")
        print()

        metrics_list.append(
            [
                degree,
                train_mse,
                test_mse,
                train_variance,
                test_variance,
                train_std,
                test_std,
            ]
        )

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_degree = degree
            best_model = model

    print(f"Best Degree: {best_degree} with Test MSE: {best_test_mse}")

    best_model.save_model("model_weights.npy")

    with open(f"figures/regression_metrics_{regularisation_type}.txt", "w") as f:
        f.write(
            "Degree\tTrain MSE\tTest MSE\tTrain Variance\tTest Variance\tTrain Std Dev\tTest Std Dev\n"
        )
        for metrics in metrics_list:
            f.write("\t".join(map(str, metrics)) + "\n")

    # Plot degree vs test MSE
    degrees = [metrics[0] for metrics in metrics_list]
    test_mses = [metrics[2] for metrics in metrics_list]

    plt.figure(figsize=(10, 6))
    plt.plot(degrees, test_mses, marker="o", linestyle="-", color="b", label="Test MSE")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Mean Squared Error (Test)")
    plt.title("Polynomial Degree vs. Test MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figures/degree_vs_test_mse_{regularisation_type}.png")


def main():

    run_on_dataset("../../data/external/linreg.csv")
    run_on_dataset(
        "../../data/external/regularisation.csv", regularisation_type="Without"
    )

    run_on_dataset(
        "../../data/external/regularisation.csv",
        regularisation_param=0.1,
        regularisation_type="l1",
    )
    run_on_dataset(
        "../../data/external/regularisation.csv",
        regularisation_param=0.1,
        regularisation_type="l2",
    )

    seeds = [42, 0, 123, 999, 2024]
    seeds_experiment(seeds, "../../data/external/linreg.csv")

    rates = [1e-6, 1e-4, 0.01, 1]
    learning_rates_experiment(rates, "../../data/external/linreg.csv")


if __name__ == "__main__":
    main()
