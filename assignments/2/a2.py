import pandas as pd
import numpy as np
import time

from a2_kmeans import *
from a2_gmm import *
from a2_pca import *
from a2_hc import *
from a2_knn import *

# Load the dataset
data = pd.read_feather("../../data/external/word-embeddings.feather")
X = np.stack(data["vit"].values)
words = np.stack(data["words"].values)


#######################################################################
X = reduce_data_to_n(X, 512)

print(data.head())
kmeans1 = elbow_method(X)
result = perform_k_means(X, kmeans1)

print(f"No. of clusters: {kmeans1}")
print(f"Cluster assignments: {result['clusters']}")

print(f"Within-Cluster Sum of Squares (WCSS): {result['wcss']}")
plot_kmeans_clusters(
    X, clusters=result["clusters"], k=kmeans1, name="figures/kmeans1.png"
)
plot_kmeans_clusters_with_words(
    X, result["clusters"], words=words, k=kmeans1, name="figures/kmeans1_words.png"
)

#######################################################################

reduced_data_50 = reduce_data_to_n(X, 50)
res2 = perform_gmm_sklearn(reduced_data_50, n_components=5)
res = perform_gmm(reduced_data_50, n_components=5)
cluster_assignments = np.argmax(res["memberships"], axis=1)

plot_gmm_clusters(reduced_data_50, cluster_assignments)
plot_gmm_clusters(reduced_data_50, res2["labels"])
print(res["likelihood"])
print(res2["likelihood"])

result = find_optimal_clusters_gmm(X, max_clusters=10)
print(result)
kgmm1 = result["optimal_bic_clusters"]
params = perform_gmm_sklearn(X, n_components=1)

print("Weights:", params["weights"])
print("Means:", params["means"])
print("Covariances:", params["covariances"])
plot_gmm_clusters(
    X, cluster_assignments=params["labels"], name="figures/gmm_kgmm1.png", words=words
)


#######################################################################

reduced_data_2d, reduced_data_3d = perform_pca_2_3(X)


k2 = int(input("Enter optimal no. of clusters based on observation: "))
result = perform_k_means(reduced_data_2d, k2)

plot_kmeans_clusters(reduced_data_2d, result["clusters"], k2)
plot_kmeans_clusters_with_words(reduced_data_2d, result["clusters"], words, k2)

result = perform_k_means(X, k2)

plot_kmeans_clusters_with_words(
    X, result["clusters"], words, k2, name="figures/512_k2.png"
)


#######################################################################

reduced_data = reduce_data(X)

kmeans3 = elbow_method(reduced_data)
result = perform_k_means(reduced_data, kmeans3)


print(f"No. of clusters: {kmeans3}")
print(f"Cluster assignments: {result['clusters']}")
print(f"Within-Cluster Sum of Squares (WCSS): {result['wcss']}")

plot_kmeans_clusters(reduced_data, result["clusters"], 6)

plot_kmeans_clusters_with_words(
    reduced_data, result["clusters"], words=words, k=6, name="figures/words_kmeas3.png"
)

#######################################################################

print("2D data")
result = perform_gmm(reduced_data_2d, n_components=k2)
cluster_assignments = np.argmax(result["memberships"], axis=1)
print(result["likelihood"])
plot_gmm_clusters(reduced_data_2d, cluster_assignments, name="figures/gmm_k2.png")
print(cluster_assignments)
plot_gmm_clusters(
    reduced_data_2d,
    cluster_assignments,
    name="figures/gmm_k2_words.png",
    words=words,
)
print("2D data on 512")
result = perform_gmm(X, n_components=k2)
cluster_assignments = np.argmax(result["memberships"], axis=1)
print(result["likelihood"])
plot_gmm_clusters(X, cluster_assignments, name="figures/gmm_k2_whole.png")
plot_gmm_clusters(
    X, cluster_assignments, name="figures/gmm_k2_whole_words.png", words=words
)

print(cluster_assignments)


res = find_optimal_clusters_gmm_2(reduced_data)
print(res)
kgmm3 = res["optimal_bic_clusters"]

result = perform_gmm(reduced_data, kgmm3)
print(result["likelihood"])

cluster_assignments = np.argmax(result["memberships"], axis=1)
plot_gmm_clusters(
    reduced_data, cluster_assignments, name="figures/gmm_kgmm3_words.png", words=words
)


# gmm_on_toy_dataset()

#######################################################################


heirarchical_clustering(X)
hc_analysis(X=X, metric="euclidean", method="ward", words=words)
#######################################################################


df = pd.read_csv("../../data/interim/1/spotify.csv")
X_spotify = df.drop(columns=["track_genre"]).values

y = df["track_genre"].values


reduced_data = reduce_data(X_spotify, name="figures/spotify_reduced.png")
perform_knn_analysis(X_spotify, reduced_data, y)

measure_inference_time(X_spotify, reduced_data, y)
