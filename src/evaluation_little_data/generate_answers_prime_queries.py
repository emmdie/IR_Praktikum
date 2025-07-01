import pathlib
import re
import sys
import os
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).parents[2].as_posix()
sys.path.append(PROJECT_ROOT)

PATH_TO_DOC_DATA = os.path.join(PROJECT_ROOT, 'data/wikipedia/testdata/raw')
SAVING_LOC = os.path.join(PROJECT_ROOT, 'src/evaluation_little_data')

from src.static_approach.inverted_index import build_inverted_index
from src.static_approach.load_docs import *
from src.static_approach.saving_and_loading import *

if __name__ == "__main__":

    pickle_files = load_pickle_files(PATH_TO_DOC_DATA)
    # print(*pickle_files, sep='\n')
    
    category_pattern = rf'{PATH_TO_DOC_DATA}/(\w+)\.pkl\.gzip'
    for file in pickle_files:
        doc_data = pd.read_pickle(file, compression='gzip')
        category = re.search(category_pattern, file).group(1)
        docs_in_category = set(doc_data.index)
        with open(f'{SAVING_LOC}/results_{category}.txt', 'w') as results_file:
            for doc_id in docs_in_category:
                results_file.write(f"{doc_id}\n")