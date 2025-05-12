import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

repo_path = (os.path.abspath(os.path.join("", os.pardir)))
jaguar_path = os.path.join(repo_path, "Data", "JaguarTestData")
hammer_path = os.path.join(repo_path, "Data", "HammerTestData")

model = SentenceTransformer("all-mpnet-base-v2")

def load_documents(*dirs):
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

def diversify_results(embeddings, paths, texts, similarity_threshold=0.8, max_results=10):
    assert len(embeddings) == len(paths) == len(texts)
    
    selected = []
    selected_paths = []
    selected_texts = []
    
    for i in range(len(embeddings)):
        if len(selected) == 0:
            selected.append(embeddings[i])
            selected_paths.append(paths[i])
            selected_texts.append(texts[i])
        else:
            sims = [util.cos_sim(embeddings[i], e).item() for e in selected]
            if max(sims) < similarity_threshold:
                selected.append(embeddings[i])
                selected_paths.append(paths[i])
                selected_texts.append(texts[i])
        if len(selected) >= max_results:
            break
    
    print(f"\nDiversified Top-{max_results} Results:\n")
    for path, text in zip(selected_paths, selected_texts):
        print(f"{path}\n{text}\n")

    return selected_paths, selected_texts

if __name__ == "__main__":
    documents, paths = load_documents(jaguar_path, hammer_path)
    doc_embeddings = embed_documents(documents, model)

    query = "weapon"
    top_embeddings, top_paths, top_texts = retrieve_top_k(query, documents, doc_embeddings, paths, k=10)
    diversify_results(top_embeddings, top_paths, top_texts, similarity_threshold=0.9, max_results=10)
