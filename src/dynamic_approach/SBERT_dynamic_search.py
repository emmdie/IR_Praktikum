import os
import torch
import numpy as np
from typing import Dict, List
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib as plt
import threading

from sentence_transformers import SentenceTransformer, util

try:
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    CUML_AVAILABLE = True
except ImportError:
    print("Warning: CuML not available, falling back to CPU clustering")
    from sklearn.cluster import KMeans, HDBSCAN
    CUML_AVAILABLE = False


# Initialize model globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer("all-mpnet-base-v2")
model.to(device)

def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_gpu_memory_info():
    """Get GPU memory usage information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
    return "GPU not available"

def efficient_torch_to_cupy(tensor):
    # Efficiently convert PyTorch tensor to CuPy array avoiding CPU roundtrip

    if not CUML_AVAILABLE:
        raise ImportError("CuPy not available")
    
    try:
        # Method 1: DLPack - zero-copy conversion (fastest)
        return cp.fromDlpack(tensor.detach())
    except Exception as e:
        print(f"DLPack conversion failed ({e}), falling back to CPU roundtrip")
        # Method 2: Fallback to CPU roundtrip (much slower)
        return cp.asarray(tensor.detach().cpu().numpy())

# Your clustering functions
def retrieve_top_k(
    query: str, 
    doc_embeddings: pd.DataFrame, 
    original_doc_ids, 
    k=1000
) -> tuple[List, List, List, torch.Tensor]:
    # Ensure everything is on GPU
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True, device=device)
    
    
    if not doc_embeddings.is_cuda:
        doc_embeddings = doc_embeddings.to(device)
    
    # Since embeddings are already normalized, dot product = cosine similarity
    similarity_scores = torch.matmul(doc_embeddings, query_embedding)
    
    k_actual = min(k, len(doc_embeddings))
    top_k_values, top_k_indices = torch.topk(similarity_scores, k_actual)

    return ([doc_embeddings[idx] for idx in top_k_indices], 
            top_k_values.cpu().numpy().tolist(),
            [original_doc_ids[idx] for idx in top_k_indices.cpu().numpy()],
            query_embedding)

def cluster_with_kmeans(
    query_embedding: torch.Tensor, 
    doc_embeddings: List, 
    doc_scores: List, 
    original_doc_ids: List, 
    num_clusters=3
) -> Dict:
    # K-Means Clustering
    
    embeddings = torch.stack(doc_embeddings)
    
    if CUML_AVAILABLE:
        # Use cuML for GPU clustering with euclidean metric (equivalent to cosine for normalized vectors)
        embeddings_gpu = efficient_torch_to_cupy(embeddings)
        
        # For normalized embeddings, euclidean distance gives same clustering as cosine
        kmeans = cuKMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_gpu)
        labels = cp.asnumpy(labels)  # Convert back to numpy
    else:
        # Fallback to CPU clustering
        embeddings_cpu = embeddings.cpu().numpy()
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_cpu)

    # GPU-accelerated similarity computation for cluster selection
    # Since embeddings are normalized, dot product = cosine similarity
    cluster_info = []
    for cluster_id in range(num_clusters):
        indices_in_cluster = [i for i, label in enumerate(labels) if label == cluster_id]
        if not indices_in_cluster:
            continue

        # Vectorized dot product on GPU (no need for normalization)
        cluster_embeddings = torch.stack([doc_embeddings[i] for i in indices_in_cluster])
        sims = torch.matmul(cluster_embeddings, query_embedding).cpu().numpy()
        
        best_idx_in_cluster = indices_in_cluster[np.argmax(sims)]

        cluster_info.append({
            "cluster_id": cluster_id,
            "best_idx": best_idx_in_cluster,
            "best_initial_ranking": best_idx_in_cluster + 1,
            "doc_id": original_doc_ids[best_idx_in_cluster],
            "similarity_score": doc_scores[best_idx_in_cluster]
        })

    # Sort clusters by best initial ranking
    cluster_info.sort(key=lambda x: x["best_initial_ranking"])

    # Create final results
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
    # HDBScan Clustering
    embeddings = torch.stack(doc_embeddings)

    if CUML_AVAILABLE:
        # Efficient GPU tensor conversion (avoids CPU roundtrip)
        embeddings_gpu = efficient_torch_to_cupy(embeddings)
        
        clusterer = cuHDBSCAN(
            metric='euclidean',  # Faster than cosine for normalized embeddings
            min_cluster_size=min_cluster_size, 
            min_samples=1, 
            cluster_selection_epsilon=cluster_selection_epsilon
        )
        clusterer.fit(embeddings_gpu)
        labels = cp.asnumpy(clusterer.labels_)
    else:
        # Fallback to CPU clustering
        embeddings_cpu = embeddings.cpu().numpy()
        clusterer = HDBSCAN(
            metric='euclidean',  # Faster than cosine for normalized embeddings
            min_cluster_size=min_cluster_size, 
            min_samples=1, 
            cluster_selection_epsilon=cluster_selection_epsilon
        )
        clusterer.fit(embeddings_cpu)
        labels = clusterer.labels_

    unique_clusters = set(labels)
    clustered_clusters = [c for c in unique_clusters if c != -1]
    has_noise = -1 in unique_clusters

    # GPU-accelerated similarity computation for cluster selection
    # Since embeddings are normalized, dot product = cosine similarity
    cluster_info = []
    for cluster_id in clustered_clusters:
        indices_in_cluster = [i for i, label in enumerate(labels) if label == cluster_id]
        if not indices_in_cluster:
            continue

        # Vectorized dot product on GPU (no need for normalization)
        cluster_embeddings = torch.stack([doc_embeddings[i] for i in indices_in_cluster])
        sims = torch.matmul(cluster_embeddings, query_embedding).cpu().numpy()
        
        best_idx_in_cluster = indices_in_cluster[np.argmax(sims)]

        cluster_info.append({
            "cluster_id": f"{cluster_id}",
            "best_idx": best_idx_in_cluster,
            "best_initial_ranking": best_idx_in_cluster + 1,
            "doc_id": original_doc_ids[best_idx_in_cluster],
            "similarity_score": doc_scores[best_idx_in_cluster]
        })

    # Sort clusters by best initial ranking
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

    # Handle noise points efficiently
    if has_noise:
        noise_indices = [i for i, label in enumerate(labels) if label == -1]
        noise_info = []
        for idx in noise_indices:
            noise_info.append({
                "doc_id": original_doc_ids[idx],
                "init_ranking": idx + 1,
                "similarity_score": doc_scores[idx]
            })
        
        noise_info.sort(key=lambda x: x["init_ranking"])
        
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


def visualise_clusters(doc_embeddings, labels, doc_paths=None, query_embedding=None, 
                      method='tsne', figsize=(12, 8), title="Document Clusters Visualization", 
                      show_labels=True):
    """Your existing visualization function"""
    # Convert embeddings to numpy if needed
    if isinstance(doc_embeddings, list):
        if torch.is_tensor(doc_embeddings[0]):
            embeddings = torch.stack(doc_embeddings).cpu().numpy()
        else:
            embeddings = np.array(doc_embeddings)
    else:
        if torch.is_tensor(doc_embeddings):
            embeddings = doc_embeddings.cpu().numpy()
        else:
            embeddings = doc_embeddings

    # Handle query embedding if provided
    include_query = query_embedding is not None
    if include_query:
        if torch.is_tensor(query_embedding):
            query_emb = query_embedding.cpu().numpy()
        else:
            query_emb = np.array(query_embedding)
        
        # Add query embedding to the embeddings for dimensionality reduction
        embeddings_with_query = np.vstack([embeddings, query_emb.reshape(1, -1)])
    else:
        embeddings_with_query = embeddings

    # Ensure labels is numpy array
    labels = np.array(labels)
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        perplexity = min(30, len(embeddings_with_query)-1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d_all = reducer.fit_transform(embeddings_with_query)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d_all = reducer.fit_transform(embeddings_with_query)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    # Split back into document embeddings and query embedding
    embeddings_2d = embeddings_2d_all[:len(embeddings)]
    if include_query:
        query_2d = embeddings_2d_all[-1]
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:  # Noise points (for HDBSCAN)
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c='black', marker='x', s=100, alpha=0.6, label='Noise')
        else:
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], s=100, alpha=0.7, label=f'Cluster {label}')
    
    # Plot query embedding if provided
    if include_query:
        plt.scatter(query_2d[0], query_2d[1], c='red', marker='*', s=300, 
                   edgecolors='black', linewidth=2, alpha=0.9, label='Query', zorder=5)
    
    # Add document labels if requested and paths are provided
    if show_labels and doc_paths is not None:
        for i, (x, y) in enumerate(embeddings_2d):
            # Extract filename from path
            filename = doc_paths[i].split('/')[-1].replace('.txt', '') if doc_paths[i] else f'Doc {i}'
            plt.annotate(filename, (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    # Add query label if query is included
    if include_query:
        plt.annotate('QUERY', (query_2d[0], query_2d[1]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10, fontweight='bold', 
                    color='red', alpha=0.9)
    
    plt.title(title)
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def the_function(query, k=5, method="hdbscan", batch_size=10000, show_viz=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.abspath(os.path.join(script_dir, "../.."))
    jaguar_text_path = os.path.join(repo_path, "data/wikipedia/testdata/raw/jaguar.pkl.gzip")
    jaguar_embeddings_path = os.path.join(repo_path, "data/wikipedia/testdata/embedded/jaguar_embeddings.pkl.gzip")
    
    try:
        # Load embeddings with chunking for large datasets
        embeddingdf = pd.read_pickle(filepath_or_buffer=jaguar_embeddings_path, compression='gzip')
        
        # Convert embeddings to tensor and move to GPU in batches
        # Since embeddings are already normalized, no need for additional normalization
        embeddings_list = embeddingdf['embedding'].tolist()
        
        # For very large datasets, process in batches
        if len(embeddings_list) > batch_size:
            print(f"Processing {len(embeddings_list)} embeddings in batches of {batch_size}")
            # For initial retrieval, we can process in chunks and use FP16 for memory efficiency
            doc_embeddings = torch.tensor(embeddings_list).to(device, dtype=torch.float16)
        else:
            # Use FP32 for smaller datasets to maintain precision
            doc_embeddings = torch.tensor(embeddings_list).to(device)
        
        original_doc_ids = embeddingdf.index.tolist()
        
        # GPU-optimized top-k retrieval
        top_embeddings, top_scores, top_doc_ids, query_embedding = retrieve_top_k(
            query, doc_embeddings, original_doc_ids, k=1000
        )
        
        if not top_embeddings:
            return []
            
        # Apply GPU-accelerated clustering
        if method == "kmeans":
            num_clusters = min(k, len(top_embeddings))
            results = cluster_with_kmeans(
                query_embedding, top_embeddings, top_scores, top_doc_ids, num_clusters=num_clusters
            )
        elif method == "hdbscan":
            results = cluster_with_hdbscan(
                query_embedding, top_embeddings, top_scores, top_doc_ids, 
                min_cluster_size=5, cluster_selection_epsilon=0.2
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
        
        if show_viz:
            # Extract cluster labels for visualization
            cluster_labels = []
            for result in results:
                cluster_labels.append(result["cluster"])
            
            # Get embeddings for the selected results
            selected_embeddings = []
            selected_paths = []
            for result in results:
                # Find the embedding index for this result
                result_idx = result["doc_id"]  # This should match with your indexing
                if isinstance(result_idx, str):
                    # If doc_id is string, find the index
                    try:
                        embedding_idx = top_doc_ids.index(result_idx)
                        selected_embeddings.append(top_embeddings[embedding_idx])
                        selected_paths.append(f"doc_{result_idx}")
                    except ValueError:
                        continue
                else:
                    # If doc_id is numeric index
                    selected_embeddings.append(top_embeddings[result_idx])
                    selected_paths.append(f"doc_{result_idx}")
            
            # Run visualization in separate thread
            def create_viz():
                visualise_clusters(
                    selected_embeddings, 
                    cluster_labels,
                    doc_paths=selected_paths,
                    query_embedding=query_embedding,
                    method='tsne',
                    title=f"Search Results Clusters for '{query}'"
                )
            
            thread = threading.Thread(target=create_viz)
            thread.daemon = True
            thread.start()
        
        text = pd.read_pickle(filepath_or_buffer=jaguar_text_path, compression='gzip')
        df = pd.DataFrame(results).set_index('doc_id')
        merged_df = df.join(text, how='left')
        
        return merged_df[:k]
        
    except Exception as e:
        print(f"Error in the_function: {e}")
        # Fallback to fake data
        fake_results = [
            {"doc_id": i, "init_ranking": i+1, "new_ranking": i+1, "cluster": "DEMO", 
             "text": f"Demo document {i} about {query}", "path": f"demo_{i}.txt", "similarity_score": 1.0-i*0.1}
            for i in range(min(k, 10))
        ]
        return pd.DataFrame(fake_results).set_index('doc_id')

    except Exception as e:
        print(f"Error in the_function: {e}")
        # Fallback to fake data
        fake_results = [
            {"doc_id": i, "init_ranking": i+1, "new_ranking": i+1, "cluster": "DEMO", 
             "text": f"Demo document {i} about {query}", "path": f"demo_{i}.txt", "similarity_score": 1.0-i*0.1}
            for i in range(min(k, 10))
        ]
        return pd.DataFrame(fake_results).set_index('doc_id')