import torch
import pandas as pd
import pathlib
from sentence_transformers import util, SentenceTransformer
from typing import Dict, List, Any

from saving_and_loading import load_pickle
from load_docs import load_doc_data, load_doc_embeddings

model = SentenceTransformer("all-mpnet-base-v2")

# currently exact match
def find_category(query: str, categories: Any) -> str:
    """
    Determine the matching category for a query.
    For now, exact matching is used.
    """
    return query

def retrieve_top_k(
    query_embedding: torch.Tensor,
    doc_embeddings: List[torch.Tensor],
    doc_ids: List[str],
    k: int = 3
) -> List[str]:
    """
    Retrieve top-k most similar document IDs based on cosine similarity.
    """
    doc_embeddings = torch.stack(doc_embeddings)
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_k_indices = torch.topk(cos_scores, k=k).indices
    
    return [doc_ids[i] for i in top_k_indices]

def search(
    query: str,
    df_doc_emb: pd.DataFrame,
    representatives: Dict[str, Dict[int, Any]],
    k: int = 3
) -> pd.DataFrame:
    """
    Run semantic search using cluster centroids and return top-matching documents.

    Args:
        query (str): User query string.
        df_doc_emb (pd.DataFrame): DataFrame of document embeddings.
        representatives (dict): Mapping from category → cluster → centroid embedding.
        k (int): Number of top documents to retrieve per cluster.

    Returns:
        pd.DataFrame: Ranked document matches with category and cluster metadata.
    """
    # Find category
    category = find_category(query, representatives.keys())
    query_semantics = representatives[category]
    
    # Load all embeddings
    doc_embeddings = list(map(torch.Tensor, df_doc_emb.embedding))
    doc_ids = df_doc_emb.index.tolist()
    
    results = []

    # NOTE Documents may appear in multiple clusters due to overlapping semantics

    # retrieve top_k documents for each semantic
    for cluster_label, query_semantic_embedding in query_semantics.items():
        top_ids = retrieve_top_k(query_semantic_embedding, doc_embeddings, doc_ids, k)
        for doc_id in top_ids:
            results.append({
                'd_id': doc_id,
                'category': category,
                'cluster': cluster_label
            })
    results_df = pd.DataFrame(results).set_index('d_id')
    results_df = rank(results_df, query, df_doc_emb)
    return results_df

def rank(results_df: pd.DataFrame, query: str, df_doc_emb) -> pd.DataFrame:
    """
    Rank matched documents by cosine similarity to the query.

    Args:
        results_df (pd.DataFrame): DataFrame of matched documents.
        query (str): User query string.
        df_doc_emb (pd.DataFrame): DataFrame of document embeddings.

    Returns:
        pd.DataFrame: Ranked results with 'rank' column (1 = most similar).
    """

    # Compute embedding of query and get embeddings of results
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True) # TODO check if this matches the embedding config used
    doc_ids = results_df.index.tolist()
    results_embeddings = [torch.tensor(df_doc_emb.loc[doc_id].embedding) for doc_id in doc_ids]
    
    # Rank documents based on cosine similarity to query
    cos_scores = util.cos_sim(query_embedding, torch.stack(results_embeddings))[0]
    results_df = results_df.copy()
    results_df['new_rank'] = cos_scores.argsort(descending=True).argsort().add(1).numpy()
    
    return results_df

def add_doc_texts(results_df: pd.DataFrame, df_doc_data: pd.DataFrame) -> pd.DataFrame:
    """
    Join the result DataFrame with the document texts based on the document ID.
    """
    return results_df.join(df_doc_data.text)

def sbert_static_search(
    query: str = "hammer",
    path_to_doc_data: str = "data/wikipedia/testdata/raw", 
    path_to_doc_emb: str = "data/test-data-martin", 
    path_to_representatives: str = "data/static-approach"
) -> pd.DataFrame:
    """
    Execute the static SBERT-based semantic search.

    Steps:
    - Load precomputed semantic cluster representatives
    - Run semantic search
    - Join document texts
    """ 
    project_root = pathlib.Path(__file__).parents[2]
    
    path_to_doc_data = (project_root / path_to_doc_data).as_posix()
    path_to_doc_emb = (project_root / path_to_doc_emb).as_posix()
    path_to_representatives = (project_root / path_to_representatives).as_posix()
    
    df_doc_data = load_doc_data(path_to_doc_data)
    df_doc_emb = load_doc_embeddings(path_to_doc_emb)
    
    
    # LOADING
    print('SEARCHING...')
    print('Loading semantics...')
    representatives_loaded = load_pickle(path_to_representatives, "representatives.pkl")
    print('Semantics loaded')

    # SEARCHING
    print('Retrieving documents...')
    search_results = search(query, df_doc_emb, representatives_loaded)
    search_results = add_doc_texts(search_results, df_doc_data)
    print(f'Search results: {search_results}')

    print(f'Number of clusters: {len(search_results)}')

    return search_results

if __name__ == "__main__":
    
    sbert_static_search("jaguar")
