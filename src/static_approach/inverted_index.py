import re
from collections import defaultdict

def tokenize(string : str):
    tokens = set(re.findall(r'\w+', string.lower()))
    return tokens

def build_inverted_index(df_text):
    word_to_strings = defaultdict(set)
    for d_id, row in df_text.iterrows():
        row_string = f'{row.label} {row.text}'
        for word in tokenize(row_string):
            word_to_strings[word].add(d_id)
    return word_to_strings