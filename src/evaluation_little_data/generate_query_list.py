import pathlib
import sys
import os

PROJECT_ROOT = pathlib.Path(__file__).parents[2].as_posix()
sys.path.append(PROJECT_ROOT)

PATH_TO_DOC_DATA = os.path.join(PROJECT_ROOT, 'data/wikipedia/testdata/raw')
QUERIES_SAVING_LOC = os.path.join(PROJECT_ROOT, 'src/evaluation_little_data')

from src.static_approach.inverted_index import build_inverted_index
from src.static_approach.load_docs import *
from src.static_approach.saving_and_loading import *

CATEGORY_RELEVANCE_THRESHOLD = 100

if __name__ == "__main__":

    doc_data = load_doc_data(PATH_TO_DOC_DATA)

    categories_dict = build_inverted_index(doc_data)
    categories = set(categories_dict.keys())
    categories = {cat for cat in categories if len(categories_dict[cat]) > CATEGORY_RELEVANCE_THRESHOLD}
 
    saving_string = os.path.join(QUERIES_SAVING_LOC, "queries.txt")
    with open(saving_string, 'w') as query_file:
        for cat in categories:
            query_file.write(f"{cat}\n")

