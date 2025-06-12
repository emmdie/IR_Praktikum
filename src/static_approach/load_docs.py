import os, glob
import re, pandas as pd

def get_categories_from_filenames(pickle_files):

    file_names = [os.path.basename(file) for file in pickle_files]

    category_names = [ re.sub(r'_embeddings.pkl$', r'', name) for name in file_names]

    return category_names

def load_docs():
    path_to_file = os.path.dirname(os.path.abspath(__file__))

    path_to_data = os.path.abspath(os.path.join(path_to_file, '../../data/test-data-martin'))

    pickle_files = glob.glob(os.path.join(path_to_data,'*.pkl'))

    categories = get_categories_from_filenames(pickle_files)

    df_list = [pd.read_pickle(file).assign(label=label) for file, label in zip(pickle_files, categories)]

    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df
    
if __name__ == "__main__":

    df = load_docs()

    print(df)