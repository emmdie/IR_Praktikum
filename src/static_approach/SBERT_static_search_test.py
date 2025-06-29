from SBERT_static_search import sbert_static_search

"""
    This is a minimal working example of the static approach
    It uses 
        - a dict containing the representatives (semantics)
        - a dataframe containing the doc_embedding
        - and a dataframe containing the remaining data
    The files are loaded on import. Their directories are hardcoded within SBERT_static_search.py.
    Either change them yourself or pass the loaded data yourself (see SBERT_static_search_hpc.py)
"""
if __name__ == "__main__":
    
    query = input("Enter a query!\n")
    
    while query not in ["quit", "q"]:
        results = sbert_static_search(query=query, num_docs_to_retrieve=10)
        print(results)
        query = input("Enter a query!")