from load_docs import *
from typing import List, Dict, Set
import pickle
import pandas as pd
from collections import defaultdict
import re

def split_dict_into_batches(d: Dict, n_batches: int) -> List[Dict]:
    items = list(d.items())
    return [dict(items[i::n_batches]) for i in range(n_batches)]

#### inverted index #############
def tokenize(string : str, translation_table):
    # Remove dots and commas to improve abbreviation support
    string = string.translate(translation_table)

    tokens = set(re.findall(r'\w+', string.lower())[:35])
    return tokens

def build_inverted_index(df_text : pd.DataFrame) -> Dict[str, Set[str]]:
    word_to_strings = defaultdict(set)
    symbols_to_remove = '.,'
    translation_table = str.maketrans('', '', symbols_to_remove)
    for i, (d_id, row) in enumerate(df_text.iterrows()):
        row_string = f'{row.label} {row.text}'
        for word in tokenize(row_string, translation_table):
            word_to_strings[word].add(d_id)
        if i == 5000:
            return word_to_strings
    return word_to_strings

def compute_categories(docs: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Build an inverted index mapping each category to its associated document IDs.
    
    Args:
        docs (pd.DataFrame): DataFrame containing document metadata, including categories.
    
    Returns:
        dict: A mapping from category names to sets of document IDs.
    """
    return build_inverted_index(docs)

def main(doc_data_dir: str, batch_saving_location: str, n_batches: int) -> None:
    # df_doc_data = load_doc_data_hpc(doc_data_dir)
    df_doc_data = load_doc_data(doc_data_dir)
    print(df_doc_data)

    categories = compute_categories(df_doc_data)
    batches = split_dict_into_batches(categories, n_batches=n_batches)
    
    for i, batch in enumerate(batches):
        print(len(batch.items()))
        path = batch_saving_location + f"/batch_{i}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(batch, f)

if __name__ == "__main__":
    # doc_data_dir = "/home/martin/University/08_IRP/IR_Praktikum/data/wikipedia/split-data-no-disambiguation"
    batch_saving_location = "/home/martin/University/08_IRP/IR_Praktikum/data/test-data-martin/batching"
    doc_data_dir: str = "/home/martin/University/08_IRP/IR_Praktikum/data/wikipedia/testdata/raw"
    n_batches = 20
    main(doc_data_dir, batch_saving_location, n_batches)