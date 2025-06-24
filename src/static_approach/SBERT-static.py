import os, sys, uuid
from collections import defaultdict
import torch, numpy as np
from sentence_transformers import SentenceTransformer, util
import show

# Add the parent directory of 'static_approach' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.clustering_methods import HDBClustering
from src.static_approach.saving_and_loading import save_representatives_pickle, load_representatives_pickle
import src.read_files
from load_docs import load_doc_data, load_doc_embeddings
from inverted_index import build_inverted_index

model = SentenceTransformer("all-mpnet-base-v2")

class Document:

    def __init__(self, id, category, embedding, content):
        self.id = id
        self.category = category
        self.content = content
        self.embedding = embedding
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
repo_path = os.path.abspath(os.path.join(script_dir, "../../data"))

def create_doc_collection(doc_collections, repo_path):
    docs = dict()
    for doc_collection in doc_collections:
        path = os.path.join(repo_path, doc_collection)
        doc_texts, _ = src.read_files.load_documents(path)
        for doc_text in doc_texts:
            doc_obj = Document(uuid.uuid4(), doc_collection, doc_text)
            docs[doc_obj.id] = doc_obj
    return docs

def create_doc_collection_from_df(df):
    docs = dict()
    for i, (index, row) in enumerate(df.iterrows()):
        doc_obj = Document(row.d_id, row.label, row.embedding, '')
        if i == 10:
            break
    return docs

def compute_categories(docs):
    return build_inverted_index(docs)

def embed(docs, model):
    # Extract the contents from the document dict in insertion order
    contents = [doc.content for doc in docs.values()]

    # Generate embeddings for the entire batch
    embeddings = model.encode(contents, convert_to_tensor=True, normalize_embeddings=True)

    # Assign the embeddings back to the respective documents in the dictionary
    for idx, (doc_id, doc) in enumerate(docs.items()):
        doc.embedding = embeddings[idx]

    return list(zip([doc.id for doc in docs.values()], embeddings))

def compute_clustering(df_doc_emb, categories):
    clustering = dict()
    for ctr, (category, doc_ids_in_category) in enumerate(categories.items()):
        if len(doc_ids_in_category) > 8000:
            print(f'{ctr:} {category} SKIPPED')
            continue
        else:
            print(f'{ctr:} {category}')

        clustering[category] = defaultdict(list)
        
        docs_in_category = df_doc_emb.loc[list(doc_ids_in_category)]
        embeddings = list(map(torch.Tensor, docs_in_category.embedding))
        if len(embeddings) > 1:
            num_clusters, clusters = HDBClustering(embeddings)
        elif len(embeddings) == 1:
            clusters = [1]
        else:
            print(f'No embeddings in this category: {category}')


        if len(clusters) != len(doc_ids_in_category):
            print(f"Length of labels does not match the number of documents in category '{category}'")
            continue

        for doc_id, cluster in zip(doc_ids_in_category, clusters):
            clustering[category][cluster].append(doc_id)

        # if ctr == 10:
        #     return clustering

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

def sbert_static_load(docs):
    # categories: category -> doc_id
    categories = compute_categories(docs)

    # clustering: category -> cluster -> doc_id
    clustering = compute_clustering(docs, categories)

    # representatives: category -> embedding
    representatives = compute_representatives(docs, clustering)

    if categories.keys() != representatives.keys():
        print(f"Categories and categories in representatives do not match!")

    save_representatives_pickle(representatives)

def sbert_static_search(docs):
    ## PRE SEARCHING
    representatives_loaded = load_representatives_pickle()

    ## SEARCHING
    query = "hammer"
    search_results = search(query, docs, representatives_loaded)
    show_doc_texts(docs, search_results)

if __name__ == "__main__":
    ## LOADING
    df_doc_data = load_doc_data()
    df_doc_emb = load_doc_embeddings()

    df_doc_data = df_doc_data.sample(frac=1)

    # df_doc_data = df_doc_data.loc[list(df_doc_emb.index)]

    print(f'Length as list (data): {len(list(df_doc_data.index))}')
    print(f'Length as set (data): {len(set(df_doc_data.index))}')

    print(f'Length as list (emb): {len(list(df_doc_emb.index))}')
    print(f'Length as set (emb): {len(set(df_doc_emb.index))}')

    print(f'doc_ids equal: {set(df_doc_emb.index) == (set(df_doc_data.index))}')


    print(df_doc_data)
    print(df_doc_emb)

    # print(df_doc_data[df_doc_data.index.duplicated(keep=False)].sort_values(by='label'))
    # print(df_doc_emb[df_doc_emb.index.isna()])
    # nan = df_doc_emb[df_doc_emb['label'].isna()]

    categories = compute_categories(df_doc_data)

    # show.highest_doc_freq(categories)

    clustering = compute_clustering(df_doc_emb, categories)
    
    # TODO get overview of cluster sizes and doc_frequencies
    # TODO compute representatives

    # docs: doc_id -> doc_obj
    # TODO load docs from dataframe
    # docs = create_doc_collection(doc_collections, repo_path)

    # sbert_static_load(docs)

    # sbert_static_search(docs)