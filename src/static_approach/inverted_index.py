import re
import pandas as pd
from typing import Dict, Set
from collections import defaultdict


def tokenize(string : str):
    # Remove dots and commas to improve abbreviation support
    symbols_to_remove = '.,'
    translation_table = str.maketrans('', '', symbols_to_remove)
    string = string.translate(translation_table)

    tokens = set(re.findall(r'\w+', string.lower())[:35])
    return tokens

def build_inverted_index(df_text : pd.DataFrame) -> Dict[str, Set[str]]:
    word_to_strings = defaultdict(set)
    for i, (d_id, row) in enumerate(df_text.iterrows()):
        row_string = f'{row.label} {row.text}'
        for word in tokenize(row_string):
            word_to_strings[word].add(d_id)
        if i == 5000:
            return word_to_strings
    return word_to_strings