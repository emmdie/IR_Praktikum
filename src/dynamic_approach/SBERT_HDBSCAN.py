import os
import torch
import numpy as np
from typing import Dict, List
import pandas as pd
from pathlib import Path

from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans, HDBSCAN

# Initialize model globally
model = SentenceTransformer("all-mpnet-base-v2")

# Your clustering functions
def retrieve_top_k(query, doc_embeddings, original_doc_ids, k=1000):
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    
    # Ensure both tensors are on the same device
    device = query_embedding.device
    if hasattr(doc_embeddings, 'device') and doc_embeddings.device != device:
        doc_embeddings = doc_embeddings.to(device)
    
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_k_indices = torch.topk(cos_scores, k=min(k, len(doc_embeddings))).indices

    return ([doc_embeddings[idx] for idx in top_k_indices], 
            [cos_scores[idx].item() for idx in top_k_indices],
            [original_doc_ids[idx] for idx in top_k_indices],
            query_embedding)

def cluster_and_select_top_from_each(query_embedding, doc_embeddings, doc_scores, original_doc_ids, num_clusters=3):
    # Move to CPU for clustering
    embeddings = torch.stack(doc_embeddings).cpu().numpy()

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Collect cluster info with best initial ranking per cluster
    cluster_info = []
    for cluster_id in range(num_clusters):
        indices_in_cluster = [i for i, label in enumerate(labels) if label == cluster_id]
        if not indices_in_cluster:
            continue

        cluster_embeddings = [doc_embeddings[i] for i in indices_in_cluster]
        sims = [util.cos_sim(query_embedding, emb).item() for emb in cluster_embeddings]
        best_idx_in_cluster = indices_in_cluster[np.argmax(sims)]

        cluster_info.append({
            "cluster_id": cluster_id,
            "best_idx": best_idx_in_cluster,
            "best_initial_ranking": best_idx_in_cluster + 1,  # +1 because rankings start at 1
            "doc_id": original_doc_ids[best_idx_in_cluster],
            "similarity_score": doc_scores[best_idx_in_cluster]
        })

    # Sort clusters by best initial ranking (lower is better)
    cluster_info.sort(key=lambda x: x["best_initial_ranking"])

    # Create final results with new ranking based on cluster priority
    selected = []
    for rank, cluster in enumerate(cluster_info, 1):
        selected.append({
            "doc_id": cluster["doc_id"],
            "init_ranking": cluster["best_initial_ranking"],
            "new_ranking": rank,
            "cluster": cluster["cluster_id"],
            "similarity_score": cluster["similarity_score"]
        })

    return selected

def cluster_with_hdbscan(query_embedding, doc_embeddings, doc_scores, original_doc_ids, min_cluster_size=100, cluster_selection_epsilon=0.0):
    # Move to CPU for clustering
    embeddings = torch.stack(doc_embeddings).cpu().numpy()

    clusterer = HDBSCAN(metric='cosine', min_cluster_size=min_cluster_size, min_samples=1, cluster_selection_epsilon=cluster_selection_epsilon)
    clusterer.fit(embeddings)

    labels = clusterer.labels_
    unique_clusters = set(labels)
    
    # Separate clustered and noise points
    clustered_clusters = [c for c in unique_clusters if c != -1]
    has_noise = -1 in unique_clusters

    # Collect cluster info with best initial ranking per cluster
    cluster_info = []
    for cluster_id in clustered_clusters:
        indices_in_cluster = [i for i, label in enumerate(labels) if label == cluster_id]
        if not indices_in_cluster:
            continue

        cluster_embeddings = [doc_embeddings[i] for i in indices_in_cluster]
        sims = [util.cos_sim(query_embedding, emb).item() for emb in cluster_embeddings]
        best_idx_in_cluster = indices_in_cluster[np.argmax(sims)]

        cluster_info.append({
            "cluster_id": f"C{cluster_id}",
            "best_idx": best_idx_in_cluster,
            "best_initial_ranking": best_idx_in_cluster + 1,
            "doc_id": original_doc_ids[best_idx_in_cluster],
            "similarity_score": doc_scores[best_idx_in_cluster]
        })

    # Sort clusters by best initial ranking (lower is better)
    cluster_info.sort(key=lambda x: x["best_initial_ranking"])

    # Create results for clustered documents
    selected = []
    for rank, cluster in enumerate(cluster_info, 1):
        selected.append({
            "doc_id": cluster["doc_id"],
            "init_ranking": cluster["best_initial_ranking"],
            "new_ranking": rank,
            "cluster": cluster["cluster_id"],
            "similarity_score": cluster["similarity_score"]
        })

    # Add noise points at the end, sorted by their initial ranking
    if has_noise:
        noise_indices = [i for i, label in enumerate(labels) if label == -1]
        noise_info = []
        for idx in noise_indices:
            noise_info.append({
                "doc_id": original_doc_ids[idx],
                "init_ranking": idx + 1,
                "similarity_score": doc_scores[idx]
            })
        
        # Sort noise points by initial ranking
        noise_info.sort(key=lambda x: x["init_ranking"])
        
        # Add noise points to results
        current_rank = len(selected) + 1
        for noise in noise_info:
            selected.append({
                "doc_id": noise["doc_id"],
                "init_ranking": noise["init_ranking"],
                "new_ranking": current_rank,
                "cluster": "NOISE",
                "similarity_score": noise["similarity_score"]
            })
            current_rank += 1

    return selected

def the_function(query, k=5, method="hdbscan"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.abspath(os.path.join(script_dir, "../.."))
    jaguar_text_path = os.path.join(repo_path, "data/wikipedia/testdata/raw/jaguar.pkl.gzip")
    jaguar_embeddings_path = os.path.join(repo_path, "data/wikipedia/testdata/embedded/jaguar_embeddings.pkl.gzip")
    
    try:
        # Load the embeddings dataframe
        embeddingdf = pd.read_pickle(filepath_or_buffer=jaguar_embeddings_path, compression='gzip')
        
        # Extract embeddings and ensure they're on CPU initially
        doc_embeddings = torch.tensor(embeddingdf['embedding'].tolist()).cpu()
        original_doc_ids = embeddingdf.index.tolist()  # Get original doc_ids from index
        
        # Get top 1000 documents
        top_embeddings, top_scores, top_doc_ids, query_embedding = retrieve_top_k(
            query, doc_embeddings, original_doc_ids, k=1000
        )
        
        if not top_embeddings:
            return []
            
        # Apply clustering to get diverse results
        if method == "kmeans":
            num_clusters = min(k, len(top_embeddings))
            results = cluster_and_select_top_from_each(
                query_embedding, top_embeddings, top_scores, top_doc_ids, num_clusters=num_clusters
            )
        elif method == "hdbscan":
            results = cluster_with_hdbscan(
                query_embedding, top_embeddings, top_scores, top_doc_ids, min_cluster_size=5, cluster_selection_epsilon=0.2
            )
        else:  # original
            results = [
                {
                    "doc_id": top_doc_ids[i],
                    "init_ranking": i + 1,
                    "new_ranking": i + 1,
                    "cluster": "N/A",
                    "similarity_score": top_scores[i]
                }
                for i in range(min(k, len(top_scores)))
            ]
            
        text = pd.read_pickle(filepath_or_buffer=jaguar_text_path, compression='gzip')
        df = pd.DataFrame(results).set_index('doc_id')
        merged_df = df.join(text, how='left')
        
        return merged_df[:k]  # Return top k results
        
    except Exception as e:
        print(f"Error in the_function: {e}")
        # Fallback to fake data
        fake_results = [
            {"doc_id": i, "init_ranking": i+1, "new_ranking": i+1, "cluster": "DEMO", 
             "text": f"Demo document {i} about {query}", "path": f"demo_{i}.txt", "similarity_score": 1.0-i*0.1}
            for i in range(min(k, 10))
        ]
        return pd.DataFrame(fake_results).set_index('doc_id')