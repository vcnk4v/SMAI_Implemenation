import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath("../../"))

from models.k_means.k_means import KMeans


def single_kmeans(X, k):
    kmeans = KMeans(k=k)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    cost = kmeans.getCost()

    print(f"No. of clusters: {k}")
    print(f"Cluster assignments: {clusters}")
    print(f"Within-Cluster Sum of Squares (WCSS): {cost}")


def perform_k_means(X, k):
    kmeans = KMeans(k=k)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    cost = kmeans.getCost()

    print(f"No. of clusters: {k}")
    print(f"Cluster assignments: {clusters}")
    print(f"Within-Cluster Sum of Squares (WCSS): {cost}")

    return {"clusters": clusters, "wcss": cost}


def plot_kmeans_clusters_with_words(X, clusters, words, k, name=None):
    cmap = plt.cm.get_cmap("viridis", k)

    plt.figure(figsize=(10, 8))

    for cluster in range(k):
        plt.scatter(
            X[clusters == cluster, 0],
            X[clusters == cluster, 1],
            label=f"Cluster {cluster}",
            color=cmap(cluster),  # Get color from 'viridis' colormap
        )

    # Add the words to the plot
    for i, word in enumerate(words):
        plt.annotate(word, (X[i, 0], X[i, 1]), fontsize=9, alpha=0.75)

    plt.title(f"K-means Clustering with {k} Clusters and Words")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    # plt.show()
    if name:
        plt.savefig(name)
    else:
        plt.savefig(f"kmeans_words_{X.shape[1]}.png")


def plot_kmeans_clusters(X, clusters, k, name=None):
    cmap = plt.cm.get_cmap("viridis", k)

    plt.figure(figsize=(10, 8))

    for cluster in range(k):
        plt.scatter(
            X[clusters == cluster, 0],
            X[clusters == cluster, 1],
            label=f"Cluster {cluster}",
            color=cmap(cluster),  # Get color from 'viridis' colormap
        )

    plt.title(f"K-means Clustering with {k} Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    if name:
        plt.savefig(name)
    else:
        # plt.show()
        plt.savefig(f"kmeans_{X.shape[1]}.png")


def elbow_method(X):
    wcss_values = []
    for k in range(1, 11):
        wcss_values.append(perform_k_means(X, k)["wcss"])

    plt.figure()
    plt.plot(range(1, 11), wcss_values, marker="o")
    plt.xlabel("Clusters Count")
    plt.ylabel("WCSS")
    # plt.show()
    plt.savefig("figures/elbow_kmeans1.png")

    kmeans1 = int(input("Enter optimal number of clusters: "))

    return kmeans1
