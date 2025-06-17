import torch, pandas as pd
from sentence_transformers import util
from saving_and_loading import load_pickle
from load_docs import load_doc_data, load_doc_embeddings

# currently exact match
def find_category(query, categories):
    return query

def retrieve_top_k(query_embedding, doc_embeddings, doc_ids, k=3):
    doc_embeddings = torch.stack(doc_embeddings)
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_k_indices = torch.topk(cos_scores, k=k).indices
    return [doc_ids[i] for i in top_k_indices]

def search(query, df_doc_emb, representatives, k=3):
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

def add_doc_texts(results_df : pd.DataFrame, df_doc_data : pd.DataFrame) -> pd.DataFrame:
    return results_df.join(df_doc_data.text)

def sbert_static_search(df_doc_data : pd.DataFrame, df_doc_emb : pd.DataFrame) -> pd.DataFrame:
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