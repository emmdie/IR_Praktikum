import os
from SBERT_static_search import sbert_static_search, HPC_TESTING
from saving_and_loading import load_pickle_gz
from load_docs import load_doc_data, load_doc_embeddings

"""
    This script is designated for evaluation on the hpc.
    It supports 
        - a list of queries to be executed
        - a blueprint how to use loaded data stored in a location different to the default location
"""

if __name__ == "__main__":
    HPC_TESTING = True

    PWD = os.getcwd() # current working directory - or any other path prefix you'd like to use
    
    # Paths to the files - need to be absolute
    # In this example are computed based on PWD
    path_to_doc_data = os.path.join(PWD, "data/wikipedia/split-data-no-disambiguation")
    path_to_doc_emb = os.path.join(PWD, "../new_embeddings")
    path_to_representatives = os.path.join(PWD, "data/representatives")

    # Loading data
    print("Loading data for SBERT static search...")
    representatives_loaded = load_pickle_gz(path_to_representatives, "representatives.pkl.gz")
    df_doc_data = load_doc_data(path_to_doc_data)
    df_doc_emb = load_doc_embeddings(path_to_doc_emb)
    print("Data loaded.")
    
    # Search
    queries = ["hi", "jaguar", "hammer"]

    for query in queries:
        results = sbert_static_search(
            query=query, 
            num_docs_to_retrieve=100,
            doc_data=df_doc_data,
            doc_emb=df_doc_emb,
            representatives=representatives_loaded
            )
        print(results)



