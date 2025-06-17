import torch
import pandas as pd
from sentence_transformers import util
from typing import Dict, List, Any

from saving_and_loading import load_pickle
from load_docs import load_doc_data, load_doc_embeddings


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
    Run semantic search over clustered document embeddings.

    Returns:
        A DataFrame containing matched document IDs, their category, and cluster label.
    """
    category = find_category(query, representatives.keys())
    query_semantics = representatives[category]
    
    doc_embeddings = list(map(torch.Tensor, df_doc_emb.embedding))
    doc_ids = df_doc_emb.index.tolist()
    
    results = []

    # IT CAN HAPPEN that a document is found with different semantics and therefore is contained in several clusters!

    for cluster_label, query_semantic_embedding in query_semantics.items():
        top_ids = retrieve_top_k(query_semantic_embedding, doc_embeddings, doc_ids, k)
        for doc_id in top_ids:
            results.append({
                'd_id': doc_id,
                'category': category,
                'cluster': cluster_label
            })
    results_df = pd.DataFrame(results).set_index('d_id')
    # TODO initial Ordering
    return results_df

def add_doc_texts(results_df: pd.DataFrame, df_doc_data: pd.DataFrame) -> pd.DataFrame:
    """
    Join the result DataFrame with the document texts based on the document ID.
    """
    return results_df.join(df_doc_data.text)

def sbert_static_search(df_doc_data: pd.DataFrame, df_doc_emb: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the static SBERT-based semantic search.

    Steps:
    - Load precomputed semantic cluster representatives
    - Run semantic search
    - Join document texts
    """
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

    sbert_static_search(df_doc_data, df_doc_emb)