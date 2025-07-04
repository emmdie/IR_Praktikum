import os
import glob
import pandas as pd

# NOTE Works with both path being absolute or relative to this script-file, because of load_pickle_files

# Works both with absolute and relative path, because of behavior of os.path.join()
def load_pickle_files(path):
    
    # Compute directory path, i.e. make absolute if it is relative - works also if <path> absolute
    path_to_file = os.path.dirname(os.path.abspath(__file__))
    path_to_data = os.path.abspath(os.path.join(path_to_file, path))

    # Compute file paths
    pickle_files_gzip = glob.glob(os.path.join(path_to_data, '*.pkl.gzip'))
    pickle_files_gz = glob.glob(os.path.join(path_to_data, '*.pkl.gz'))
    pickle_files = pickle_files_gzip + pickle_files_gz

    return pickle_files

def pandas_load_df_from_pickle(path):
    
    # Load pickle files
    pickle_files = load_pickle_files(path)    
    df_list = [pd.read_pickle(file, compression='gzip') for file in pickle_files]

    # Concat and eliminate duplicates (in case overlapping collections of documents are used)
    combined_df = pd.concat(df_list)
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    return combined_df


def load_doc_embeddings(path):
    return pandas_load_df_from_pickle(path)

def load_doc_data(path):
    return pandas_load_df_from_pickle(path)
    


