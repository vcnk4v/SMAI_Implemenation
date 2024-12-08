import scipy.cluster.hierarchy as hc
import matplotlib.pyplot as plt


def heirarchical_clustering(X):
    linkage_methods = [
        "single",
        "complete",
        "average",
        "weighted",
        "centroid",
        "median",
        "ward",
    ]
    distance_metrics = ["euclidean", "cityblock", "cosine"]

    for method in linkage_methods:
        for metric in distance_metrics:
            # Skip incompatible combinations
            if metric != "euclidean" and method in ["centroid", "median", "ward"]:
                continue

            # Compute the linkage matrix directly
            Z = hc.linkage(X, method=method, metric=metric)

            # Plot the dendrogram
            plt.figure(figsize=(10, 7))
            hc.dendrogram(Z)
            plt.title(f"Dendrogram - Linkage: {method}, Distance Metric: {metric}")
            plt.xlabel("Data Points")
            plt.ylabel("Distance")
            # plt.show()
            plt.savefig(f"figures/hc_{metric}_{method}.png")


def hc_analysis(method, metric, words, X):
    # Compute the linkage matrix
    Z = hc.linkage(X, method=method, metric=metric)

    # Define the number of clusters to evaluate
    kbest1 = 3
    kbest2 = 6

    # Perform clustering
    clusters_hier_kbest1 = hc.fcluster(Z, t=kbest1, criterion="maxclust")
    clusters_hier_kbest2 = hc.fcluster(Z, t=kbest2, criterion="maxclust")

    # Helper function to print clusters
    def print_clusters(clusters, k):
        print(f"\nWords grouped into {k} clusters:")
        for cluster in range(1, k + 1):
            cluster_words = [
                words[i] for i in range(len(words)) if clusters[i] == cluster
            ]
            print(f"  Cluster {cluster}: {', '.join(cluster_words)}")

    # Print clusters for both kbest1 and kbest2
    print_clusters(clusters_hier_kbest1, kbest1)
    print_clusters(clusters_hier_kbest2, kbest2)
