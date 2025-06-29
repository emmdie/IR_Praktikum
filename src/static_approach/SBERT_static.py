import os, sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import show
from SBERT_static_load import sbert_static_load
from SBERT_static_search import sbert_static_search

if __name__ == "__main__":
    """
        Configuration is using default values of function arguments. 
        Ensure that the files are in the respective directories or pass the directories where your files are located.
    """
    # Given documents and it's embeddings 
    # compute the semantics of each word found in the documents and save it (representatives.pkl)
    sbert_static_load()    
    
    # Search a given document collection given the documents themselves and their embeddings 
    search_results = sbert_static_search(query="hammer",num_docs_to_retrieve=20)

    show.doc_texts_clusterwise(search_results)

