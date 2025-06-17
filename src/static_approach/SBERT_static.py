import os, sys
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.static_approach.saving_and_loading import load_pickle
from load_docs import load_doc_data, load_doc_embeddings
import show
from SBERT_static_load import sbert_static_load
from SBERT_static_search import sbert_static_search

model = SentenceTransformer("all-mpnet-base-v2")


if __name__ == "__main__":

    df_doc_data = load_doc_data()
    df_doc_emb = load_doc_embeddings()

    print('Finished loading data')

    sbert_static_load(df_doc_data, df_doc_emb)    
    
    # search_results = sbert_static_search(df_doc_data, df_doc_emb)

    # show.doc_texts_clusterwise(search_results)

    # number_of_duplicates = len(search_results.index[search_results.index.duplicated()].unique())

    # print(f'Exists document in several clusters: {search_results.index.has_duplicates}')
    # print(f'Number of duplicates: {number_of_duplicates}')
    
