import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.abspath(os.path.join(script_dir, ".."))
jaguar_path = os.path.join(repo_path, "data", "JaguarTestData")
hammer_path = os.path.join(repo_path, "Data", "HammerTestData")

model = SentenceTransformer("all-mpnet-base-v2")

def load_documents(*dirs): #Nimmt beliebig viele Ordner mit txt files
    doc_texts = []
    doc_paths = []
    for dir_path in dirs:
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    doc_texts.append(content)
                    doc_paths.append(file_path)
    return doc_texts, doc_paths

def embed_documents(docs, model):
    return model.encode(docs, convert_to_tensor=True, normalize_embeddings=True)

def retrieve_top_k(query, doc_texts, doc_embeddings, doc_paths, k=3):
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_k_indices = torch.topk(cos_scores, k=k).indices

    print(f"Top {k} documents for query: '{query}'\n")
    for idx in top_k_indices:
        print(f"[Score: {cos_scores[idx]:.4f}] {doc_paths[idx]}\n{doc_texts[idx]}\n")
    
    return [doc_embeddings[idx] for idx in top_k_indices], [doc_paths[idx] for idx in top_k_indices], [doc_texts[idx] for idx in top_k_indices]

#Erstellt statische Anzahl an Clustern, das funktioniert gut
def cluster_and_select_top_from_each(query_embedding, doc_embeddings, doc_texts, doc_paths, num_clusters=3):
    embeddings = torch.stack(doc_embeddings).numpy()

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    selected = []
    for cluster_id in range(num_clusters):
        indices_in_cluster = [i for i, label in enumerate(labels) if label == cluster_id]
        if not indices_in_cluster:
            continue

        cluster_embeddings = [doc_embeddings[i] for i in indices_in_cluster]
        sims = [util.cos_sim(query_embedding, emb).item() for emb in cluster_embeddings]
        best_idx_in_cluster = indices_in_cluster[np.argmax(sims)]

        cluster_other_paths = [doc_paths[i] for i in indices_in_cluster if i != best_idx_in_cluster]

        selected.append({
            "path": doc_paths[best_idx_in_cluster],
            "text": doc_texts[best_idx_in_cluster],
            "others": cluster_other_paths
        })

    print(f"\nCluster-Diversified Results (1 per cluster):\n")
    for item in selected:
        print(f"{item['path']}\n{item['text']}")
        if item["others"]:
            print(f"# Cluster also includes: {', '.join(item['others'])}")
        print()

    return selected

#Erstellt dynamische Anzahl an Clustern, funktioniert weniger gut
def cluster_with_hdbscan(query_embedding, doc_embeddings, doc_texts, doc_paths, min_cluster_size=2):
    # Convert to numpy array for clustering
    embeddings = torch.stack(doc_embeddings).numpy()

    # Cluster using HDBScan
    clusterer = HDBSCAN(metric='cosine', min_cluster_size=min_cluster_size, min_samples=1)
    clusterer.fit(embeddings)

    # Eliminate the unclustered elements
    labels = clusterer.labels_
    unique_clusters = set(labels)
    if -1 in unique_clusters:
        unique_clusters.remove(-1) # -1 is the noise cluster

    # Collect information of elements in clusters
    selected = []
    for cluster_id in unique_clusters:
        indices_in_cluster = [i for i, label in enumerate(labels) if label == cluster_id]
        if not indices_in_cluster:
            continue

        cluster_embeddings = [doc_embeddings[i] for i in indices_in_cluster]
        sims = [util.cos_sim(query_embedding, emb).item() for emb in cluster_embeddings]
        best_idx_in_cluster = indices_in_cluster[np.argmax(sims)]

        cluster_other_paths = [doc_paths[i] for i in indices_in_cluster if i != best_idx_in_cluster]

        selected.append({
            "path": doc_paths[best_idx_in_cluster],
            "text": doc_texts[best_idx_in_cluster],
            "others": cluster_other_paths
        })

    # Collect the unclustered elements
    noise_indices = [i for i, label in enumerate(labels) if label == -1]
    print("Noise Cluster Elements:")
    for i in noise_indices:
        print(f"- Path: {doc_paths[i]}")
        print(f"  Text: {doc_texts[i][:200]}...\n")

    return selected


def visualise_clusters(doc_embeddings, labels, doc_paths=None, query_embedding=None, 
                      method='tsne', figsize=(12, 8), title="Document Clusters Visualization", 
                      show_labels=True):
    """
    Visualize document embeddings in 2D space colored by cluster labels.
    
    Args:
        doc_embeddings: List of torch tensors or numpy array of embeddings
        labels: List/array of cluster labels for each document
        doc_paths: Optional list of document paths for hover/legend info
        query_embedding: Optional query embedding to include in visualization
        method: Dimensionality reduction method ('tsne' or 'pca')
        figsize: Figure size tuple
        title: Plot title
        show_labels: Whether to show document names as text labels
    """
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

# Modified versions of your clustering functions to return labels for visualization
def cluster_and_select_with_labels(query_embedding, doc_embeddings, doc_texts, doc_paths, num_clusters=3):
    """Modified version that returns cluster labels for visualization"""
    embeddings = torch.stack(doc_embeddings).cpu().numpy()
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Your existing selection logic
    selected = []
    for cluster_id in range(num_clusters):
        indices_in_cluster = [i for i, label in enumerate(labels) if label == cluster_id]
        if not indices_in_cluster:
            continue

        cluster_embeddings = [doc_embeddings[i] for i in indices_in_cluster]
        sims = [util.cos_sim(query_embedding, emb).item() for emb in cluster_embeddings]
        best_idx_in_cluster = indices_in_cluster[np.argmax(sims)]

        cluster_other_paths = [doc_paths[i] for i in indices_in_cluster if i != best_idx_in_cluster]

        selected.append({
            "path": doc_paths[best_idx_in_cluster],
            "text": doc_texts[best_idx_in_cluster],
            "others": cluster_other_paths
        })

    print(f"\nCluster-Diversified Results (1 per cluster):\n")
    for item in selected:
        print(f"{item['path']}\n{item['text']}")
        if item["others"]:
            print(f"# Cluster also includes: {', '.join(item['others'])}")
        print()

    return selected, labels

def cluster_with_hdbscan_labels(query_embedding, doc_embeddings, doc_texts, doc_paths, min_cluster_size=2):
    """Modified version that returns cluster labels for visualization"""
    # Convert to numpy array for clustering
    embeddings = torch.stack(doc_embeddings).cpu().numpy()

    # Cluster using HDBScan
    clusterer = HDBSCAN(metric='cosine', min_cluster_size=min_cluster_size, min_samples=1)
    labels = clusterer.fit_predict(embeddings)

    # Your existing selection logic
    unique_clusters = set(labels)
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    selected = []
    for cluster_id in unique_clusters:
        indices_in_cluster = [i for i, label in enumerate(labels) if label == cluster_id]
        if not indices_in_cluster:
            continue

        cluster_embeddings = [doc_embeddings[i] for i in indices_in_cluster]
        sims = [util.cos_sim(query_embedding, emb).item() for emb in cluster_embeddings]
        best_idx_in_cluster = indices_in_cluster[np.argmax(sims)]

        cluster_other_paths = [doc_paths[i] for i in indices_in_cluster if i != best_idx_in_cluster]

        selected.append({
            "path": doc_paths[best_idx_in_cluster],
            "text": doc_texts[best_idx_in_cluster],
            "others": cluster_other_paths
        })

    noise_indices = [i for i, label in enumerate(labels) if label == -1]
    print("Noise Cluster Elements:")
    for i in noise_indices:
        print(f"- Path: {doc_paths[i]}")
        print(f"  Text: {doc_texts[i][:200]}...\n")

    return selected, labels

# Example usage in your main function:
if __name__ == "__main__":
    documents, paths = load_documents(jaguar_path, hammer_path)
    doc_embeddings = embed_documents(documents, model)
    query = "hammer"
    query_embedding = model.encode(query, convert_to_tensor=True)
    top_embeddings, top_paths, top_texts = retrieve_top_k(query, documents, doc_embeddings, paths, k=10)
    
    # Use the modified clustering function that returns labels
    selected, labels = cluster_with_hdbscan_labels(
        query_embedding=query_embedding,
        doc_embeddings=top_embeddings,
        doc_texts=top_texts,
        doc_paths=top_paths
    )
    
    # Visualize the clusters
    visualise_clusters(top_embeddings, labels, top_paths, query_embedding=query_embedding, 
                      method='pca', title="HDBSCAN Clustering of Document Embeddings")
    
'''
Evaluation
- eine Finale Zahl um darzustellen wie gut ein Datensatz ist (TFIDF vs BERT)
- in top 10 wie viele Cluster sind vertreten?
- Statische Themenbereiche durch Disambiguation Pages
- F1 Maß
- 

TODO:
- Martin TestDataensatz mit embeddings? (df: doc_id, embedding)
- Reiner vielleicht Freitag/Samstag/Sontag
- Emmanuel Freitag/Samstag
'''