from sklearn.cluster import HDBSCAN
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

def KMeansClustering(num_clusters, doc_embeddings):
    embeddings = torch.stack(doc_embeddings).numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return num_clusters, labels


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

def HDBClustering(doc_embeddings, min_cluster_size=2):
    # Convert to numpy array for clustering
    embeddings = torch.stack(doc_embeddings).numpy()

    # Cluster using HDBScan
    clusterer = HDBSCAN(metric='cosine', min_cluster_size=min_cluster_size, min_samples=1)
    clusterer.fit(embeddings)

    # Eliminate the unclustered elements
    labels = clusterer.labels_
    unique_clusters = set(labels)
    if -1 in unique_clusters:
        unique_clusters.remove(-1) # -1 is the noise cluster
    num_clusters = len(unique_clusters)
    return num_clusters, labels