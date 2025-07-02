from sklearn.cluster import HDBSCAN, MiniBatchKMeans
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# Config variables
from config_load import *

if CUML:
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans
    from cuml.cluster import HDBSCAN as cuHDBSCAN


def KMeansClustering(doc_embeddings: np.ndarray, num_clusters: int):
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(doc_embeddings)
    return labels, num_clusters

def cuKMeansClustering(doc_embeddings: np.ndarray, num_clusters: int):
    kmeans = cuKMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(doc_embeddings)
    labels = cp.asnumpy(labels)  # Convert back to numpy
    return labels, num_clusters

def MiniBatchKMeansClustering(
        doc_embeddings: np.ndarray, 
        num_clusters: int, 
        batch_size=102,
        init_size=None,
        max_iter=100,
        reassignment_ratio=0.01,
        random_state=None
    ):
    mini_batch_kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=batch_size,
        init_size=init_size,
        max_iter=max_iter,
        reassignment_ratio=reassignment_ratio,
        random_state=random_state
    )
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

def cuHDBClustering(doc_embeddings: np.ndarray, metric: str = 'cosine', cluster_selection_epsilon=0.0, alpha=1.0, min_cluster_size: int=2,):

    # min cluster size is no less then two
    min_cluster_size = max(2, min_cluster_size)
    
    # Cluster using HDBScan
    clusterer = cuHDBSCAN(metric=metric, min_cluster_size=min_cluster_size, min_samples=1, cluster_selection_epsilon=cluster_selection_epsilon, alpha=alpha)
    clusterer.fit(doc_embeddings)

    # Eliminate the unclustered elements
    labels = cp.asnumpy(clusterer.labels_)
    num_clusters = len(set(labels))
    return num_clusters, labels