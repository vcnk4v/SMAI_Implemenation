import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../../"))
from models.pca.pca import PCA


def perform_pca_2_3(X):

    # Reduce to 2 dimensions
    pca_2d = PCA(n_components=2)
    pca_2d.fit(X)
    reduced_data_2d = pca_2d.transform(X)
    is_valid_2d = pca_2d.checkPCA(X)
    print("2D PCA check:", is_valid_2d)

    # Reduce to 3 dimensions
    pca_3d = PCA(n_components=3)
    pca_3d.fit(X)
    reduced_data_3d = pca_3d.transform(X)
    is_valid_3d = pca_3d.checkPCA(X)
    print("3D PCA check:", is_valid_3d)

    plt.figure(figsize=(10, 7))
    plt.scatter(
        reduced_data_2d[:, 0], reduced_data_2d[:, 1], c="blue", edgecolor="k", s=50
    )
    plt.title("PCA - 2D")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    # plt.show()
    plt.savefig("figures/pca_2d.png")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        reduced_data_3d[:, 0],
        reduced_data_3d[:, 1],
        reduced_data_3d[:, 2],
        c="red",
        edgecolor="k",
        s=50,
    )
    ax.set_title("PCA - 3D")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    # plt.show()
    plt.savefig("figures/pca_3d.png")

    ## k2 = 5?
    return reduced_data_2d, reduced_data_3d


def reduce_data(X, name=None):

    pca_full = PCA()
    pca_full.fit(X)

    # Calculate the explained variance ratio
    explained_variance_ratio = pca_full.explained_variance / np.sum(
        pca_full.explained_variance
    )
    plt.figure(figsize=(10, 6))
    plt.plot(
        (np.arange(1, len(explained_variance_ratio) + 1)),
        explained_variance_ratio,
        marker="o",
        linestyle="--",
    )
    plt.title("Scree Plot")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Explained Variance Ratio")
    # plt.show()
    plt.savefig("figures/scree_plot_512.png")

    # Generate scree plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        (np.arange(1, len(explained_variance_ratio) + 1))[:50],
        explained_variance_ratio[:50],
        marker="o",
        linestyle="--",
    )
    plt.title("Scree Plot")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Explained Variance Ratio")
    # plt.show()

    if name:
        plt.savefig(name)
    else:
        plt.savefig("figures/scree_plot_50.png")
    n_optimal = int(input("Enter optimal no of components"))

    # Perform dimensionality reduction
    pca_optimal = PCA(n_components=n_optimal)
    pca_optimal.fit(X)
    reduced_data = pca_optimal.transform(X)

    print(f"Reduced data shape: {reduced_data.shape}")

    return reduced_data


def reduce_data_to_n(X, n):

    # Perform dimensionality reduction
    pca_optimal = PCA(n_components=n)
    pca_optimal.fit(X)
    reduced_data = pca_optimal.transform(X)

    print(f"Reduced data shape: {reduced_data.shape}")

    return reduced_data
