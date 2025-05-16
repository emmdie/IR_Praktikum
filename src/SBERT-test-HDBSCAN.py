import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.abspath(os.path.join(script_dir, ".."))
jaguar_path = os.path.join(repo_path, "Data", "JaguarTestData")
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


if __name__ == "__main__":
   documents, paths = load_documents(jaguar_path, hammer_path)
   doc_embeddings = embed_documents(documents, model)
   query = "hammer"
   query_embedding = model.encode(query, convert_to_tensor=True)
   top_embeddings, top_paths, top_texts = retrieve_top_k(query, documents, doc_embeddings, paths, k=10)
cluster_with_hdbscan(
    query_embedding=query_embedding,
    doc_embeddings=top_embeddings,
    doc_texts=top_texts,
    doc_paths=top_paths
) 
