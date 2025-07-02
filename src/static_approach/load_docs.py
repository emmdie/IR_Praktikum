import os
import glob
import pandas as pd

"""
    This file is dedicated to specifically load the dataframes containing
        - the embeddings for all documents
        - general data of all documents
"""

# NOTE Works with both path being absolute or relative to this script-file, because of load_pickle_files

# Works both with absolute and relative path, because of behavior of os.path.join()
def load_pickle_files(path):
    
    # Compute directory path, i.e. make absolute if it is relative - works also if <path> absolute
    path_to_file = os.path.dirname(os.path.abspath(__file__))
    path_to_data = os.path.abspath(os.path.join(path_to_file, path))

    path_to_data=path
    print(f"Trying to load data from this directory: {path_to_data}")
    # Compute file paths
    pickle_files_gzip = glob.glob(os.path.join(path_to_data, '*.pkl.gzip'))
    pickle_files_gz = glob.glob(os.path.join(path_to_data, '*.pkl.gz'))
    pickle_files = pickle_files_gzip + pickle_files_gz

    return pickle_files

def pandas_load_df_from_pickle(path):
    
    # Load pickle files
    pickle_files = load_pickle_files(path)
    
    print(f"Loading these pickle files:")
    print(*pickle_files, sep='\n')
    
    if not pickle_files:
        return None

    try:
        df_list = [pd.read_pickle(file, compression='gzip') for file in pickle_files]

        # Concat and eliminate duplicates (in case overlapping collections of documents are used)
        combined_df = pd.concat(df_list)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        print(f"Loaded these files:\n {combined_df}")

        return combined_df
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None


def load_doc_embeddings(path):
    print("Loading doc_embeddings")
    return pandas_load_df_from_pickle(path)

def load_doc_data(path):
    print("Loading doc_data")
    return pandas_load_df_from_pickle(path)
    


