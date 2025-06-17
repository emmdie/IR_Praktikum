import os, sys
from collections import defaultdict
import torch, numpy as np, pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from inverted_index import build_inverted_index
from src.clustering_methods import HDBClustering
from saving_and_loading import save_pickle
from load_docs import load_doc_data, load_doc_embeddings

def compute_categories(docs):
    return build_inverted_index(docs)

def compute_clustering(df_doc_emb, categories):
        
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

# in each category set centroid as representative for each cluster
def compute_representatives(df_doc_emb, clustering):
    representatives = dict()
    for category in clustering:
        
        representatives[category] = dict()
        clusters = clustering[category]
        for cluster in clusters:
            embeddings_in_cluster = [df_doc_emb.loc[doc_id].embedding for doc_id in clustering[category][cluster]]
            centroid = np.mean(embeddings_in_cluster, axis=0)
            representatives[category][cluster] = centroid

    return representatives

def sbert_static_load(df_doc_data : pd.DataFrame, df_doc_emb : pd.DataFrame) -> None:
    print('Starting loading phase')   
    
    # categories: category -> [doc_id]
    categories = compute_categories(df_doc_data)

    # clustering: category -> cluster -> [doc_id]
    clustering = compute_clustering(df_doc_emb, categories)

    # print('Saving/loading clustering')
    # save_pickle(clustering, "clustering.pkl")
    # clustering = load_pickle("clustering.pkl")
    # print('Finished saving/loading clustering')

    # representatives: category -> cluster_label -> embedding
    representatives = compute_representatives(df_doc_emb, clustering)

    if clustering.keys() != representatives.keys():
        print(f"Categories and categories in representatives do not match!")
        
    save_pickle(representatives)

    print('Loading Phase finished')


if __name__ == "__main__":

    df_doc_data = load_doc_data()
    df_doc_emb = load_doc_embeddings()

    print('Finished loading data')

    sbert_static_load(df_doc_data, df_doc_emb)