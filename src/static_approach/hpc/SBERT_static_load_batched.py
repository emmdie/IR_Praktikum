import re
from typing import Dict, Set
import numpy as np
import torch
import pickle
import pandas as pd
from sklearn.cluster import HDBSCAN
from collections import defaultdict
from load_docs import *
from saving_and_loading import *
import sys

def load_batch(path_to_batch_dir: str, batch_nr: int) -> pd.DataFrame:
    path_to_batch = path_to_batch_dir + f"/batch_{batch_nr}.pkl"
    with open(path_to_batch, "rb") as batch_file:
        return pickle.load(batch_file)

### HDB Clustering #####################
def HDBClustering(doc_embeddings, min_cluster_size=2, cluster_selection_epsilon=0.0, alpha=1.0):
    # Convert to numpy array for clustering
    embeddings = torch.stack(doc_embeddings).numpy()

    # min cluster size is no less then two
    min_cluster_size = max(2, min_cluster_size)

    # Cluster using HDBScan
    clusterer = HDBSCAN(metric='cosine', min_cluster_size=min_cluster_size, min_samples=1, cluster_selection_epsilon=cluster_selection_epsilon, alpha=alpha)
    clusterer.fit(embeddings)

    # Eliminate the unclustered elements
    labels = clusterer.labels_
    num_clusters = len(set(labels))
    return num_clusters, labels

### SBERT static load ###################
def compute_clustering(df_doc_emb: pd.DataFrame, categories: Dict[str, Set[str]]) -> Dict[str, Dict[int, Set[str]]]:
    """
    Cluster documents within each category using HDBSCAN, skipping very large categories.
    
    Args:
        df_doc_emb (pd.DataFrame): DataFrame containing SBERT embeddings indexed by document ID.
        categories (dict): Mapping from category to list of document IDs.
    
    Returns:
        dict: Mapping of category → cluster label → list of document IDs.
    """
    clustering = dict()
    for ctr, (category, doc_ids_in_category) in enumerate(categories.items()):
        clustering[category] = defaultdict(list)


        if category in {"the", "is", "in", "and", "to", "a", "of", "that", "it", "on", "for", "with", "as", "was", "at", "by", "an"}:
            continue
        print(f'{ctr:} {category}')

        docs_in_category = df_doc_emb.loc[list(doc_ids_in_category)]
        embeddings = list(map(torch.Tensor, docs_in_category.embedding))
        
        # min_cluster_size * num_clusters  = len(doc_ids_in_category)
        max_num_clusters = len(doc_ids_in_category) # 10 # maximum number of clusters
        
        if len(embeddings) > 1:
            min_cluster_size = int(np.ceil(len(doc_ids_in_category) / max_num_clusters))
            cluster_selection_epsilon = 0.4
            alpha = 1
            num_clusters = 11 # something greater 10
            prev_num_clusters = num_clusters + 1
            stuck_counter = 0
            first_iteration = True
            while num_clusters > 10:
                num_clusters, clusters = HDBClustering(embeddings, min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon, alpha=alpha)
                print(f'Num clusters "{category}": {num_clusters}')
                if num_clusters > 100:
                    cluster_selection_epsilon += 0.1
                elif num_clusters < 20:
                    cluster_selection_epsilon += 0.01
                else:
                    cluster_selection_epsilon += 0.03
                
                # Detect stagnation
                if num_clusters == prev_num_clusters and not first_iteration:
                    stuck_counter += 1
                else:
                    stuck_counter = 0

                if stuck_counter >= 1:
                    alpha += 0.1
                    stuck_counter = 0
                    # input(f"Alpha increased to {alpha}!")
                    # print(f"Alpha increased to {alpha}!")
                
                # If giant jump in num_cluster (like 18 to 3), undo reducing alpha
                # A giant jump only occurred after alpha reduction for values < 20
                if num_clusters <= 7 and prev_num_clusters - num_clusters >= 7 and not first_iteration:
                    # Don't reduce alpha, but epsilon instead
                    alpha -= 0.1
                    
                    # Reset cluster number
                    num_clusters = prev_num_clusters
                    prev_num_clusters += 1
                    # input(f"Undoing giant jump, using alpha = {alpha}, epsilon = {cluster_selection_epsilon}")
                else:
                    prev_num_clusters = num_clusters                
                    
                first_iteration = False
        elif len(embeddings) == 1:
            clusters = [1]
        else:
            print(f'No embeddings in this category: {category}')

        if len(clusters) != len(doc_ids_in_category):
            print(f"Length of labels does not match the number of documents in category '{category}'")
            continue

        for doc_id, cluster in zip(doc_ids_in_category, clusters):
            clustering[category][cluster].append(doc_id)

    return clustering

def compute_representatives(
    df_doc_emb: pd.DataFrame,
    clustering: Dict[str, Dict[int, Set[str]]]
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Compute centroid embedding for each cluster to use as its semantic representative.
    
    Args:
        df_doc_emb (pd.DataFrame): DataFrame of document embeddings.
        clustering (dict): Mapping from category to clusters of document IDs.
    
    Returns:
        dict: Mapping from category → cluster → centroid embedding (as numpy array).
    """
    representatives = dict()
    for category in clustering:
        
        representatives[category] = dict()
        clusters = clustering[category]
        for cluster in clusters:
            embeddings_in_cluster = [df_doc_emb.loc[doc_id].embedding for doc_id in clustering[category][cluster]]
            centroid = np.mean(embeddings_in_cluster, axis=0)
            representatives[category][cluster] = centroid

    return representatives

def sbert_static_load_hpc(
    path_to_doc_data: str,
    path_to_doc_emb: str,
    path_to_representatives: str,
    categories: pd.DataFrame,
    batch_nr: int
) -> None:
    """
    Executes the static loading pipeline:
    - Using slice of prebuilt categories dict
    - Clusters documents in each category of that slice
    - Computes centroid representatives
    - Saves the representatives to disk
    """
    print('Loading data')
    
    df_doc_data = load_doc_data(path_to_doc_data)
    df_doc_emb = load_doc_embeddings(path_to_doc_emb)

    print(df_doc_data)
    print(df_doc_emb)

    print('Finished loading data')
    
    print('Starting loading phase')   

    # 2: Cluster documents within each category
    clustering = compute_clustering(df_doc_emb, categories)

    # 3: Compute representative (centroid) embeddings per cluster
    representatives = compute_representatives(df_doc_emb, clustering)

    if clustering.keys() != representatives.keys():
        print(f"Categories and categories in representatives do not match!")
        
    # 4: Persist representatives
    save_pickle_gz(representatives, path_to_representatives, f"representatives_{batch_nr}.pkl.gz")

    print('Loading Phase finished')
        

if __name__ == "__main__":
    # path_to_doc_data = "/home/martin/University/08_IRP/IR_Praktikum/data/wikipedia/split-data-no-disambiguation"
    # path_to_doc_emb  = ""
    # path_to_representatives = "/home/martin/University/08_IRP/IR_Praktikum/data/test-data-martin/representatives"
    # path_to_batches = "/home/martin/University/08_IRP/IR_Praktikum/data/test-data-martin/batching"

    path_to_doc_data: str = "/home/martin/University/08_IRP/IR_Praktikum/data/wikipedia/testdata/raw"
    path_to_doc_emb: str = "/home/martin/University/08_IRP/IR_Praktikum/data/test-data-martin"
    path_to_batches = "/home/martin/University/08_IRP/IR_Praktikum/data/test-data-martin/batching"
    path_to_representatives = "/home/martin/University/08_IRP/IR_Praktikum/data/test-data-martin/representatives"

    i = sys.argv[1]
    categories = load_batch(path_to_batches, i)
    sbert_static_load_hpc(path_to_doc_data, path_to_doc_emb, path_to_representatives, categories, i)