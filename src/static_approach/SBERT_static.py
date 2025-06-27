import os, sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.static_approach.saving_and_loading import load_pickle
from load_docs import load_doc_data, load_doc_embeddings
import show
from SBERT_static_load import sbert_static_load
from SBERT_static_search import sbert_static_search

if __name__ == "__main__":

    df_doc_data = load_doc_data()
    df_doc_emb = load_doc_embeddings()
    print('Finished loading data')

    sbert_static_load(df_doc_data, df_doc_emb)    
    
    search_results = sbert_static_search(df_doc_data, df_doc_emb)

    show.doc_texts_clusterwise(search_results)

