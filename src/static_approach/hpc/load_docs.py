import os, glob
import re, pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


def get_categories_from_filenames(pickle_files):

    file_names = [os.path.basename(file) for file in pickle_files]

    category_names = [re.sub(r'.pkl.gzip$', r'', name) for name in file_names]

    return category_names

def load_pickle_files(path):

    path_to_file = os.path.dirname(os.path.abspath(__file__))

    path_to_data = os.path.abspath(os.path.join(path_to_file, path))

    pickle_files = glob.glob(os.path.join(path_to_data,'*.pkl.gzip'))

    pickle_files = [file for file in pickle_files if not (os.path.basename(file) == 'cat_embeddings.pkl.gzip' or os.path.basename(file) == 'cat.pkl.gzip')]
    
    return pickle_files

def load_doc_embeddings(path='../../data/test-data-martin'):
    pickle_files = load_pickle_files(path)

    df_list = [pd.read_pickle(file, compression='gzip') for file in pickle_files]

    combined_df = pd.concat(df_list)

    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    return combined_df

def load_doc_data(path='../../../data/wikipedia/testdata/raw'):
    pickle_files = load_pickle_files(path)
    
    
    categories = get_categories_from_filenames(pickle_files)

    df_list = [pd.read_pickle(file, compression='gzip').assign(category=category) for file, category in zip(pickle_files, categories)]

    combined_df = pd.concat(df_list)

    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    return combined_df

def load_doc_data_hpc(path):

    pickle_files = [path + f"/wikipedia-text-data-no-disambiguation_{i}.pkl.gzip" for i in range(13)]
    print("pickle files")
    print(*pickle_files, sep="\n")
    df_list = [pd.read_pickle(file, compression='gzip') for file in pickle_files]

    combined_df = pd.concat(df_list)

    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    print(combined_df)

    return combined_df

def load_doc_embeddings_hpc(path):
    pickle_files = [path + f"/embeddings_{i}.pkl.gzip" for i in range(13)]
    print("pickle files")
    print(*pickle_files, sep="\n")
    df_list = [pd.read_pickle(file, compression='gzip') for file in pickle_files]

    combined_df = pd.concat(df_list)

    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    print(combined_df)

    return combined_df

if __name__ == "__main__":

    load_doc_data_hpc("/home/martin/University/08_IRP/IR_Praktikum/data/wikipedia/split-data-no-disambiguation")
