import math
import os
import sys
import torch
import pandas as pd
import pathlib
from sentence_transformers import util, SentenceTransformer
from typing import Dict, List, Any, Set

static_approach_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(static_approach_dir))

from saving_and_loading import load_pickle_gz
from load_docs import load_doc_data, load_doc_embeddings
from word_matching import *
import show

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-mpnet-base-v2", device=device)

# Config
HPC_TESTING = False

# LOAD needed data
# Default paths relative to project root
DEFAULT_DOC_DATA_PATH = "data/wikipedia/testdata/raw"
DEFAULT_DOC_EMB_PATH = "data/wikipedia/testdata/embedded"
DEFAULT_REPRESENTATIVES_PATH = "data/static-approach/testing/hdbscan" #/representatives-hpc/repr_w_stopwords"

# Compute absolute paths at module import
project_root = pathlib.Path(__file__).parents[2]
path_to_doc_data = (project_root / DEFAULT_DOC_DATA_PATH).as_posix()
path_to_doc_emb = (project_root / DEFAULT_DOC_EMB_PATH).as_posix()
path_to_representatives = (project_root / DEFAULT_REPRESENTATIVES_PATH).as_posix()

# Load everything eagerly, once
# Any of this will return None if the loading has failed. This is expected behavior in sbert_static_search
print("Loading data for SBERT static search...")
representatives_loaded = load_pickle_gz(path_to_representatives, "representatives.pkl.gz") if path_to_representatives else None
df_doc_data = load_doc_data(path_to_doc_data) if path_to_doc_data else None
df_doc_emb = load_doc_embeddings(path_to_doc_emb) if path_to_doc_emb else None

if representatives_loaded and df_doc_data is not None and df_doc_emb is not None:
    print("Data loaded.")
else:
    print("Failed to print some of the data")


# currently exact match
def find_category(query: str, categories: Set[str], fuzzy_threshold=85, sim_threshold=0.6, word_embeddings_cache=None) -> str:
    """
    Determine the matching category for a query.
    For now, exact matching is used, because there is category for each token found in the training collection.
    The collection currently used is wikipedia.
    """

    # 1. Exact match
    if query in categories:
        print(f"Using exact match with '{query}'")
        return query
    
    if not HPC_TESTING:
        # 2. Fuzzy match
        fuzzy_word = fuzzy_match(query, categories, fuzzy_threshold)
        if fuzzy_word:
            print(f"Using fuzzy match with '{fuzzy_word}'")
            return fuzzy_word

        # 3. Substring fuzzy match
        for word in categories:
            candidates = set()
            if query in word:
                candidates.add(word)
            print(f"Candidates: {candidates}")
            if not candidates:
                break
            fuzzy_word = fuzzy_match(query, categories, 0)
            print(f"Using substring fuzzy match with '{fuzzy_word}'")
            return fuzzy_word

    # # 3. Embedding match # EXTREMELY LOW IN PERFORMANCE
    # semantic_word = embedding_match(query, categories, model, sim_threshold, word_embeddings_cache)
    # if semantic_word:
    #     print("Using embedding match!")
    #     return semantic_word

    print(f"Couldn't find a category for query '{query}'")
    return None

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
    num_docs_to_retrieve: int,
    exactly_retrieve_num: bool
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
    category = find_category(query, set(representatives.keys()))
    
    # Case that no category was found. This is the only case where this function might return None
    if category is None:
        return None

    query_semantics = representatives[category]
    
    # Load all embeddings
    doc_embeddings = list(map(torch.Tensor, df_doc_emb.embedding))
    doc_ids = df_doc_emb.index.tolist()
    
    results = []

    # NOTE Documents may appear in multiple clusters due to overlapping semantics
    num_categories = len(representatives[category].keys())
    num_docs_per_cluster = math.ceil(num_docs_to_retrieve / num_categories)
    num_docs_per_cluster = max(num_docs_per_cluster, 1)

    # Retrieve top_k documents for each semantic
    for cluster_label, query_semantic_embedding in query_semantics.items():
        top_ids = retrieve_top_k(query_semantic_embedding, doc_embeddings, doc_ids, k=num_docs_per_cluster)
        for doc_id in top_ids:
            results.append({
                'doc_id': doc_id,
                'cluster': cluster_label
            })
    results_df = pd.DataFrame(results).set_index('doc_id')
    results_df = rank(results_df, query, df_doc_emb)

    if exactly_retrieve_num:
        # Not only retrieve at least <num_docs_to_retrieve> documents, but exactly <num_docs_to_retrieve>
        results_df = results_df.iloc[:num_docs_to_retrieve]

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
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True, device="cpu") # TODO check if this matches the embedding config used
    doc_ids = results_df.index.tolist()
    results_embeddings = [torch.tensor(df_doc_emb.loc[doc_id].embedding) for doc_id in doc_ids]
    
    # Rank documents based on cosine similarity to query
    cos_scores = util.cos_sim(query_embedding, torch.stack(results_embeddings))[0]
    results_df = results_df.copy()
    results_df['similarity_score'] = cos_scores
    results_df['new_ranking'] = cos_scores.argsort(descending=True).argsort().add(1).numpy()

    results_df = results_df.sort_values('new_ranking')
    
    return results_df

def add_doc_texts_and_label(results_df: pd.DataFrame, df_doc_data: pd.DataFrame) -> pd.DataFrame:
    """
    Join the result DataFrame with the document texts and labels based on the document ID.
    """
    return results_df.join(df_doc_data.text).join(df_doc_data.label)

def sbert_static_search(
    query: str = "hammer",
    num_docs_to_retrieve: int = 5,
    exactly_retrieve_num: bool = True,
    doc_emb: pd.DataFrame = df_doc_emb,
    representatives: dict = representatives_loaded,
    doc_data: pd.DataFrame = df_doc_data
) -> pd.DataFrame:
    """
    Execute the static SBERT-based semantic search.

    Steps:
    - Load precomputed semantic cluster representatives
    - Run semantic search
    - Join document texts
    """ 
    if doc_emb is None:
        print("Document embeddings could not be loaded from default path and have not been provided.")
        return pd.DataFrame()

    if representatives is None:
        print("Representatives could not be loaded from default path and have not been provided.")
        return pd.DataFrame()

    if doc_data is None:
        print("Document data could not be loaded from default path and have not been provided.")
        return pd.DataFrame()

    if num_docs_to_retrieve <= 0:
        print("Number of documents to retrieve must be greater than zero!")
        return pd.DataFrame()

    search_results = search(query.lower(), doc_emb, representatives, num_docs_to_retrieve, exactly_retrieve_num)

    if search_results is None:
        return pd.DataFrame()
    else:
        search_results = add_doc_texts_and_label(search_results, doc_data)
        return search_results
    



# if __name__ == "__main__":
    
#     # EXAMPLE EXECUTION - Caution this loads the reloads the data with each query
    
#     # CHANGE THIS AS NEEDED
    
#     PWD = os.getcwd() # current working directory

#     # This way you could pass arguments on execution
#     query = sys.argv[1] # Pass the query like this python /path/to/SBERT_static_search.py <query> 

#     path_to_doc_data = os.path.join(PWD, "data/wikipedia/split-data-no-disambiguation")
#     path_to_doc_emb = os.path.join(PWD, "../new_embeddings")
#     path_to_representatives = os.path.join(PWD, "data/representatives")

#     representatives_loaded = load_pickle_gz(path_to_representatives, "representatives.pkl.gz")
#     df_doc_data = load_doc_data(path_to_doc_data)
#     df_doc_emb = load_doc_embeddings(path_to_doc_emb)
#     ##############################################################################################

#     results = sbert_static_search(
#         query=query, 
#         num_docs_to_retrieve=100, 
#         exactly_retrieve_num=True,
#         doc_data=df_doc_data,
#         doc_emb=df_doc_emb,
#         representatives=representatives_loaded
#     )
