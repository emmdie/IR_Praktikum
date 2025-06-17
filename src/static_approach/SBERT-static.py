import os, sys
from collections import defaultdict
import torch, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer, util


# Add the parent directory of 'static_approach' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.clustering_methods import HDBClustering
from src.static_approach.saving_and_loading import save_pickle, load_pickle
from load_docs import load_doc_data, load_doc_embeddings
from inverted_index import build_inverted_index
import show

model = SentenceTransformer("all-mpnet-base-v2")

# PREPROCESSING
# select documents with query term in title / (from collection (Maria), from Folder (Emmanuel)
# cluster within each term
# dict: term -> list of embeddings: representative (centroid) of cluster

# SEARCH
# Look up semantics/embeddings
# Search for each semantic

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.abspath(os.path.join(script_dir, "../../data"))

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

def find_category(query, categories):
    # matching_categories = [category for category in categories if query.lower() in category.lower()]
    # if len(matching_categories) > 1:
    #     print(f"Found more than one matching category!")
    # if len(matching_categories) == 0:
    #     print(f"No matching category found!")
    # return matching_categories[0]
    return query

def retrieve_top_k(query_embedding, doc_embeddings, doc_ids, k=3):
    doc_embeddings = torch.stack(doc_embeddings)
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_k_indices = torch.topk(cos_scores, k=k).indices
    return [doc_ids[i] for i in top_k_indices]

def search_old(query, df_doc_emb, representatives, k=3):
    category = find_category(query, representatives.keys())
    query_semantics = representatives[category]
    
    doc_embeddings = list(map(torch.Tensor, df_doc_emb.embedding))
    doc_ids = df_doc_emb.index.tolist()
    
    search_results = {
        cluster_label: retrieve_top_k(embedding, doc_embeddings, doc_ids, k)
        for cluster_label, embedding in query_semantics.items()
    }   
    return search_results

def search(query, df_doc_emb, representatives, k=3):
    category = find_category(query, representatives.keys())
    query_semantics = representatives[category]
    
    doc_embeddings = list(map(torch.Tensor, df_doc_emb.embedding))
    doc_ids = df_doc_emb.index.tolist()
    
    results = []

    for cluster_label, query_embedding in query_semantics.items():
        top_ids = retrieve_top_k(query_embedding, doc_embeddings, doc_ids, k)
        for doc_id in top_ids:
            results.append({
                'd_id': doc_id,
                'category': category,
                'cluster': cluster_label
            })
    results_df = pd.DataFrame(results).set_index('d_id')
    # TODO initial Ordering
    return results_df

def add_doc_texts(results_df : pd.DataFrame, df_doc_data : pd.DataFrame) -> pd.DataFrame:
    return results_df.join(df_doc_data.text)

def show_doc_texts_old(df_doc_data, search_results):
    for cluster, retrieved_docs in search_results.items():
        print(f"Cluster {cluster}:")
        for doc_id in retrieved_docs:
            print(df_doc_data.loc[doc_id].text)

def show_doc_texts_df(search_results_df : pd.DataFrame):
    search_results = {cluster: data for cluster, data in search_results_df.groupby('cluster_label')}
    for cluster, retrieved_docs in search_results.items():
        print(f"Cluster {cluster}:")
        for doc_id in retrieved_docs.index:
            print(df_doc_data.loc[doc_id].text)

def sbert_static_load(df_doc_data, df_doc_emb):
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

def sbert_static_search(df_doc_data, df_doc_emb):
    # LOADING
    print('SEARCHING...')
    print('Loading semnatics...')
    representatives_loaded = load_pickle()
    print('Semantics loaded')

    # SEARCHING
    print('Retrieving documents...')
    query = "hammer"
    search_results = search(query, df_doc_emb, representatives_loaded)
    search_results = add_doc_texts(search_results, df_doc_data)
    print(f'Search results: {search_results}')

    print(f'Number of clusters: {len(search_results)}')

    return search_results

if __name__ == "__main__":

    df_doc_data = load_doc_data()
    df_doc_emb = load_doc_embeddings()

    print('Finished loading data')

    sbert_static_load(df_doc_data, df_doc_emb)    
    
    sbert_static_search(df_doc_data, df_doc_emb)

    print()
