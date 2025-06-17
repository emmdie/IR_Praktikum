import re
from collections import defaultdict


def tokenize(string : str):
    tokens = set(re.findall(r'\w+', string.lower())[:35])
    return tokens

def build_inverted_index(df_text):
    word_to_strings = defaultdict(set)
    for i, (d_id, row) in enumerate(df_text.iterrows()):
        row_string = f'{row.label} {row.text}'
        for word in tokenize(row_string):
            word_to_strings[word].add(d_id)
        if i == 5000:
            return word_to_strings
    return word_to_strings