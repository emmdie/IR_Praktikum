import os
import sys
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Set
import pathlib

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from inverted_index import build_inverted_index
from clustering_methods import HDBClustering
from saving_and_loading import save_pickle_gz
from load_docs import load_doc_data, load_doc_embeddings

pca_enabled = True
stop_words_excluded = False
skip_large_categories = True

def compute_categories(docs: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Build an inverted index mapping each category to its associated document IDs.
    
    Args:
        docs (pd.DataFrame): DataFrame containing document metadata, including categories.
    
    Returns:
        dict: A mapping from category names to sets of document IDs.
    """
    return build_inverted_index(docs)

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


        if skip_large_categories and len(doc_ids_in_category) > 8000:
            print(f'{ctr:} {category} SKIPPED')
            continue

        if stop_words_excluded and category in {"the", "is", "in", "and", "to", "a", "of", "that", "it", "on", "for", "with", "as", "was", "at", "by", "an"}:
           continue
        
        print(f'{ctr:} {category}')

        docs_in_category = df_doc_emb.loc[list(doc_ids_in_category)]
        
        if pca_enabled:
            embeddings = np.array(docs_in_category.embedding_pca.tolist())
        else:
            embeddings = np.array(docs_in_category.embedding.tolist())
        
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

def sbert_static_load(
    path_to_doc_data: str = "data/wikipedia/testdata/raw", 
    path_to_doc_emb: str = "data/test-data-martin", 
    path_to_representatives: str = "data/static-approach"
) -> None:
    """
    Executes the static loading pipeline:
    - Creates categories by building inverted index
    - Clusters documents in each category
    - Computes centroid representatives
    - Saves the representatives to disk
    """
    print('Loading data')
    project_root = pathlib.Path(__file__).parents[2]
    
    path_to_doc_data = (project_root / path_to_doc_data).as_posix()
    path_to_doc_emb = (project_root / path_to_doc_emb).as_posix()
    path_to_representatives = (project_root / path_to_representatives).as_posix()
    
    df_doc_data = load_doc_data(path_to_doc_data)
    df_doc_emb = load_doc_embeddings(path_to_doc_emb)

    if pca_enabled:
        # Compute PCA for each vector - discarding normal embeddings for the sake of memory
        embeddings = np.array(df_doc_emb.embedding.tolist())
        embeddings_pca = PCA(n_components=0.95).fit_transform(embeddings)
        df_doc_emb['embedding_pca'] = embeddings_pca.tolist()
        df_doc_emb.drop('embedding', axis=1)

    print(df_doc_data)
    print(df_doc_emb)

    print('Finished loading data')
    
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
    save_pickle_gz(representatives, path_to_representatives, "representatives.pkl.gz")

    print('Loading Phase finished')


if __name__ == "__main__":

    # # If running this locally (not hpc) consider uncommenting if i == 5000: return word_to_strings

    # ##### THIS SECTION HAS BEEN INCLUDED FOR EXECUTION ON HPC CLUSTER ############################

    # # CHANGE THIS AS NEEDED

    # PWD = os.getcwd() # current working directory

    # path_to_doc_data = os.path.join(PWD, "data/wikipedia/split-data-no-disambiguation")
    # path_to_doc_emb = os.path.join(PWD, "../new_embeddings")
    # path_to_representatives = os.path.join(PWD, "data/representatives")
    # ##############################################################################################

    # sbert_static_load(
    #     path_to_doc_data=path_to_doc_data, 
    #     path_to_doc_emb=path_to_doc_emb, 
    #     path_to_representatives=path_to_representatives
    # )

    sbert_static_load()