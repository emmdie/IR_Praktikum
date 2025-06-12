import os
import uuid
from collections import defaultdict
import uuid

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from src.clustering_methods import HDBClustering
import read_files
from src.saving_and_loading import save_representatives_pickle, load_representatives_pickle

model = SentenceTransformer("all-mpnet-base-v2")

class Document:

    def __init__(self, category, content):
        self.id = uuid.uuid4()
        self.category = category
        self.content = content
        self.embedding = None
        self.label = ""

    def __repr__(self):
        return f"Document(id={self.id}, category={self.category}, content={self.content[:30]}..., embedding={self.embedding is not None})"

# PREPROCESSING
# select documents with query term in title / (from collection (Maria), from Folder (Emmanuel)
# cluster within each term
# dict: term -> list of embeddings: representative (centroid) of cluster

# SEARCH
# Look up semantics/embeddings
# Search for each semantic

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.abspath(os.path.join(script_dir, "../Data"))

def create_doc_collection(doc_collections, repo_path):
    docs = dict()
    for doc_collection in doc_collections:
        path = os.path.join(repo_path, doc_collection)
        doc_texts, _ = read_files.load_documents(path)
        for doc_text in doc_texts:
            doc_obj = Document(doc_collection, doc_text)
            docs[doc_obj.id] = doc_obj
    return docs

def compute_categories(docs):
    category = defaultdict(list)
    doc_objs = docs.values()
    for doc in doc_objs:
        category[doc.category].append(doc.id)
    return category

def embed(docs, model):
    # Extract the contents from the document dict in insertion order
    contents = [doc.content for doc in docs.values()]

    # Generate embeddings for the entire batch
    embeddings = model.encode(contents, convert_to_tensor=True, normalize_embeddings=True)

    # Assign the embeddings back to the respective documents in the dictionary
    for idx, (doc_id, doc) in enumerate(docs.items()):
        doc.embedding = embeddings[idx]

    return list(zip([doc.id for doc in docs.values()], embeddings))

def compute_clustering(docs, categories):
    clustering = dict()
    for category, doc_ids in categories.items():
        clustering[category] = defaultdict(list)

        doc_objs_in_category = [docs[obj_id] for obj_id in doc_ids]
        embeddings = [doc.embedding for doc in doc_objs_in_category]

        num_clusters, clusters = HDBClustering(embeddings)

        if len(clusters) != len(doc_objs_in_category):
            print(f"Length of labels does not match the number of documents in category '{category}'")
            continue

        for doc_obj, cluster in zip(doc_objs_in_category, clusters):
            doc_obj.label = cluster
            clustering[category][cluster].append(doc_obj.id)

    return clustering

# in each category set centroid as representative for each cluster
def compute_representatives(docs, clustering):
    representatives = dict()
    for category in clustering:
        representatives[category] = dict()
        clusters = clustering[category]
        for cluster in clusters:
            embeddings_in_cluster = [docs[doc_id].embedding for doc_id in clustering[category][cluster]]
            centroid = np.mean(embeddings_in_cluster, axis=0)
            representatives[category][cluster] = centroid
    return representatives

def find_category(query, categories):
    matching_categories = [category for category in categories if query.lower() in category.lower()]
    if len(matching_categories) > 1:
        print(f"Found more than one matching category!")
    if len(matching_categories) == 0:
        print(f"No matching category found!")
    return matching_categories[0]

def retrieve_top_k(query_embedding, doc_embeddings, doc_ids, k=3):
    doc_embeddings = torch.stack(doc_embeddings)
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_k_indices = torch.topk(cos_scores, k=k).indices
    return [doc_ids[i] for i in top_k_indices]

def search(query, docs, representatives, k=3):
    search_results = dict()
    categories = representatives.keys()
    category = find_category(query, categories)
    semantics_of_query = representatives[category]
    doc_ids, doc_embeddings = zip(*[(doc_id, doc.embedding) for doc_id, doc in docs.items()])
    for query_semantic, query_semantic_embedding in semantics_of_query.items():
        ids_top_docs = retrieve_top_k(query_semantic_embedding, doc_embeddings, doc_ids)
        search_results[query_semantic] = ids_top_docs
    return search_results

def show_doc_texts(docs, search_results):
    for cluster, retrieved_docs in search_results.items():
        print(f"Cluster {cluster}:")
        for doc_id in retrieved_docs:
            print(docs[doc_id].content)

if __name__ == "__main__":
    ## LOADING
    doc_collections = ["JaguarTestData", "HammerTestData"]

    # docs: doc_id -> doc_obj
    # TODO load docs from dataframe
    docs = create_doc_collection(doc_collections, repo_path)

    # categories: category -> doc_id
    categories = compute_categories(docs)

    # compute embedding of each doc and attach to corresponding doc_obj
    embeddings = embed(docs, model)

    # clustering: category -> cluster -> doc_id
    clustering = compute_clustering(docs, categories)

    # representatives: category -> embedding
    representatives = compute_representatives(docs, clustering)
    save_representatives_pickle(representatives)

    if categories.keys() != representatives.keys():
        print(f"Categories and categories in representatives do not match!")

    ## PRE SEARCHING
    representatives_loaded = load_representatives_pickle()

    ## SEARCHING
    query = "hammer"
    search_results = search(query, docs, representatives_loaded)
    show_doc_texts(docs, search_results)


