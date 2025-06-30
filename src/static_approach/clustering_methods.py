from sklearn.cluster import HDBSCAN, MiniBatchKMeans
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

def KMeansClustering(doc_embeddings: np.ndarray, num_clusters: int):
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(doc_embeddings)
    return labels, num_clusters

def MiniBatchKMeansClustering(doc_embeddings: np.ndarray, num_clusters: int):
    mini_batch_kmeans = MiniBatchKMeans(n_clusters=num_clusters)
    labels = mini_batch_kmeans.fit_predict(doc_embeddings)
    return labels, num_clusters

def AgglomerativeClustering(doc_embeddings, similarity_threshold=.75):
    embeddings = torch.stack(doc_embeddings).numpy()
    distance_matrix = 1 - np.inner(embeddings, embeddings)  # cosine distance = 1 - cosine sim

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=1 - similarity_threshold
    ).fit(distance_matrix)

    labels = clustering.labels_
    num_clusters = max(labels) + 1

    return num_clusters, labels

def HDBClustering(doc_embeddings: np.ndarray, metric: str = 'cosine', cluster_selection_epsilon=0.0, alpha=1.0, min_cluster_size: int=2,):

    # min cluster size is no less then two
    min_cluster_size = max(2, min_cluster_size)

    # Cluster using HDBScan
    clusterer = HDBSCAN(metric=metric, min_cluster_size=min_cluster_size, min_samples=1, cluster_selection_epsilon=cluster_selection_epsilon, alpha=alpha)
    clusterer.fit(doc_embeddings)

    # Eliminate the unclustered elements
    labels = clusterer.labels_
    num_clusters = len(set(labels))
    return num_clusters, labels