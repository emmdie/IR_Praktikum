import os
import sys
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from inverted_index import build_inverted_index
from src.clustering_methods import HDBClustering
from saving_and_loading import save_pickle
from load_docs import load_doc_data, load_doc_embeddings

def compute_categories(docs: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build an inverted index mapping each category to its associated document IDs.
    
    Args:
        docs (pd.DataFrame): DataFrame containing document metadata, including categories.
    
    Returns:
        dict: A mapping from category names to lists of document IDs.
    """
    return build_inverted_index(docs)

def compute_clustering(df_doc_emb: pd.DataFrame, categories: Dict[str, List[str]]) -> Dict[str, Dict[int, List[str]]]:
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


        if len(doc_ids_in_category) > 8000:
            print(f'{ctr:} {category} SKIPPED')
            continue
        else:
            print(f'{ctr:} {category}')

        docs_in_category = df_doc_emb.loc[list(doc_ids_in_category)]
        embeddings = list(map(torch.Tensor, docs_in_category.embedding))
        
        # min_cluster_size * num_clusters  = len(doc_ids_in_category)
        max_num_clusters = len(doc_ids_in_category) # 10 # maximum number of clusters
        
        if len(embeddings) > 1:
            min_cluster_size = int(np.ceil(len(doc_ids_in_category) / max_num_clusters))
            num_clusters, clusters = HDBClustering(embeddings, min_cluster_size)
        elif len(embeddings) == 1:
            clusters = [1]
        else:
            print(f'No embeddings in this category: {category}')

        if len(clusters) != len(doc_ids_in_category):
            print(f"Length of labels does not match the number of documents in category '{category}'")
            continue

        for doc_id, cluster in zip(doc_ids_in_category, clusters):
            clustering[category][cluster].append(doc_id)

        # if ctr == 10:
        #     return clustering

    return clustering

def compute_representatives(
    df_doc_emb: pd.DataFrame,
    clustering: Dict[str, Dict[int, List[str]]]
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

def sbert_static_load(df_doc_data: pd.DataFrame, df_doc_emb: pd.DataFrame) -> None:
    """
    Executes the static loading pipeline:
    - Creates categories by building inverted index
    - Clusters documents in each category
    - Computes centroid representatives
    - Saves the representatives to disk
    """
    print('Starting loading phase')   
    
    # 1: Build inverted index: category → [doc_ids]
    categories = compute_categories(df_doc_data)

    # 2: Cluster documents within each category
    clustering = compute_clustering(df_doc_emb, categories)

    # 3: Compute representative (centroid) embeddings per cluster
    representatives = compute_representatives(df_doc_emb, clustering)

    if clustering.keys() != representatives.keys():
        print(f"Categories and categories in representatives do not match!")
        
    # 4: Persist representatives
    save_pickle(representatives)

    print('Loading Phase finished')


if __name__ == "__main__":

    df_doc_data = load_doc_data()
    df_doc_emb = load_doc_embeddings()

    print('Finished loading data')

    sbert_static_load(df_doc_data, df_doc_emb)