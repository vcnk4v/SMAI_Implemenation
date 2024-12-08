import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd

sys.path.append(os.path.abspath("../../"))

from models.gmm.gmm import GMM


def plot_gmm_clusters(X, cluster_assignments, name=None, words=None):
    # Get the cluster assignments by taking the argmax of the memberships

    # Plot the data points and assign colors according to the cluster
    plt.figure(figsize=(10, 8))

    # Plot each cluster
    for cluster in np.unique(cluster_assignments):
        plt.scatter(
            X[cluster_assignments == cluster, 0],
            X[cluster_assignments == cluster, 1],
            label=f"Cluster {cluster}",
        )
    if words is not None:
        for i, word in enumerate(words):
            plt.annotate(word, (X[i, 0], X[i, 1]), fontsize=9, alpha=0.7)
    plt.title("GMM Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    # plt.show()
    if name:
        plt.savefig(name)


def perform_gmm_sklearn(X, n_components, max_iter=100, random_state=42):

    gmm = GaussianMixture(
        n_components=n_components, max_iter=max_iter, random_state=random_state
    )
    gmm.fit(X)

    # Get the predicted cluster labels
    labels = gmm.predict(X)

    return {
        "gmm_model": gmm,
        "labels": labels,
        "means": gmm.means_,
        "covariances": gmm.covariances_,
        "weights": gmm.weights_,
        "converged": gmm.converged_,
        "bic": gmm.bic(X),
        "aic": gmm.aic(X),
        "likelihood": gmm.score(X),
    }


def perform_gmm(X, n_components):

    try:
        gmm = GMM(n_components=n_components)
        gmm.fit(X)
        params = gmm.getParams()
        memberships = gmm.getMembership()
        likelihood = gmm.getLogLikelihood(X)
        aic = gmm.getAIC(X)
        bic = gmm.getBIC(X)
        return {
            "gmm_model": gmm,
            "params": params,
            "memberships": memberships,
            "likelihood": likelihood,
            "aic": aic,
            "bic": bic,
        }
    except Exception as e:
        print(f"There was an error: {e}")


def find_optimal_clusters_gmm_2(X, max_clusters=10):
    bic_scores = []
    aic_scores = []

    for n in range(1, max_clusters + 1):
        gmm_result = perform_gmm(X, n_components=n)

        bic_scores.append(gmm_result["bic"])
        aic_scores.append(gmm_result["aic"])

    optimal_bic_clusters = (
        bic_scores.index(min(bic_scores)) + 1
    )  # Since range starts at 1
    optimal_aic_clusters = (
        aic_scores.index(min(aic_scores)) + 1
    )  # Since range starts at 1

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), bic_scores, label="BIC", marker="o")
    plt.plot(range(1, max_clusters + 1), aic_scores, label="AIC", marker="x")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score")
    plt.title("BIC and AIC Scores for Different Number of Clusters")
    plt.legend()
    # plt.show()
    plt.savefig("figures/aic_bic.png")

    return {
        "optimal_bic_clusters": optimal_bic_clusters,
        "optimal_aic_clusters": optimal_aic_clusters,
    }


def find_optimal_clusters_gmm(X, max_clusters=10, max_iter=100, random_state=42):

    bic_scores = []
    aic_scores = []

    for n in range(1, max_clusters + 1):
        gmm_result = perform_gmm_sklearn(
            X, n_components=n, max_iter=max_iter, random_state=random_state
        )

        bic_scores.append(gmm_result["bic"])
        aic_scores.append(gmm_result["aic"])

    optimal_bic_clusters = (
        bic_scores.index(min(bic_scores)) + 1
    )  # Since range starts at 1
    optimal_aic_clusters = (
        aic_scores.index(min(aic_scores)) + 1
    )  # Since range starts at 1

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), bic_scores, label="BIC", marker="o")
    plt.plot(range(1, max_clusters + 1), aic_scores, label="AIC", marker="x")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score")
    plt.title("BIC and AIC Scores for Different Number of Clusters by sklearn")
    plt.legend()
    # plt.show()
    plt.savefig("figures/sklearn_aic_bic.png")

    return {
        "optimal_bic_clusters": optimal_bic_clusters,
        "optimal_aic_clusters": optimal_aic_clusters,
    }


# from matplotlib.colors import ListedColormap


# def gmm_on_toy_dataset():
#     # Read csv file containing (x,y,color) and convert into df
#     df = pd.read_csv("../../data/external/data.csv", header=None)

#     # Create copy with original clusters(colors)
#     df_original = df.copy()
#     df_original = df.values[1:]
#     df_original = df_original.astype(np.float32)

#     # Make numpy array out of data convverting all data points to numbers
#     df_copy = df.copy()
#     df_copy = df.values[1:, :2]
#     df_copy = df_copy.astype(np.float32)

#     gmm = GMM(n_components=3, random_state=42)
#     gmm.fit(df_copy)
#     responsibilities = gmm.getMembership()
#     print(gmm.means)

#     # Assign each point to the cluster with the highest responsibility
#     labels = np.argmax(responsibilities, axis=1)
#     print("The log likelihood of the data is: ", gmm.getLogLikelihood(df_copy))

#     # Plot the results
#     plt.figure(figsize=(14, 6))

#     plt.subplot(1, 2, 1)
#     plt.scatter(df_original[:, 0], df_original[:, 1], c=df_original[:, 2])
#     plt.title("Original Data")
#     plt.xlabel("X")
#     plt.ylabel("Y")

#     plt.subplot(1, 2, 2)
#     # Create a scatter plot for each component
#     colors = ListedColormap(
#         ["#FF6347", "#4682B4", "#32CD32"]
#     )  # Custom colors for each cluster
#     plt.scatter(
#         df_copy[:, 0],
#         df_copy[:, 1],
#         c=labels,
#         cmap=colors,
#         s=50,
#         marker="o",
#         edgecolor="k",
#     )

#     # Plot the Gaussian means as stars
#     means = gmm.getParams()["means"]
#     plt.scatter(
#         means[:, 0],
#         means[:, 1],
#         c="yellow",
#         s=150,
#         marker="*",
#         edgecolor="k",
#         label="Centroids",
#     )

#     # Add labels and title
#     plt.title("GMM Clustering own Toy Dataset")
#     plt.xlabel("X-axis")
#     plt.ylabel("Y-axis")
#     plt.legend()

#     # Show plot
#     # plt.savefig("../2/figures/gmm_toy_dataset.png")
#     plt.show()
#     ###############################################################
#     gmm = GaussianMixture(n_components=3, max_iter=100, random_state=42)
#     gmm.fit(df_copy)

#     # Get the predicted cluster labels
#     labels = gmm.predict(df_copy)

#     plt.figure(figsize=(14, 6))

#     plt.subplot(1, 2, 1)
#     plt.scatter(df_original[:, 0], df_original[:, 1], c=df_original[:, 2])
#     plt.title("Original Data")
#     plt.xlabel("X")
#     plt.ylabel("Y")

#     plt.subplot(1, 2, 2)
#     # Create a scatter plot for each component
#     colors = ListedColormap(
#         ["#FF6347", "#4682B4", "#32CD32"]
#     )  # Custom colors for each cluster
#     plt.scatter(
#         df_copy[:, 0],
#         df_copy[:, 1],
#         c=labels,
#         cmap=colors,
#         s=50,
#         marker="o",
#         edgecolor="k",
#     )
#     plt.show()

#     print(gmm.means_)

#     print("The log likelihood of the data is: ", gmm.score(df_copy))
