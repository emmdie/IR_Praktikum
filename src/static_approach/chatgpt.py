import re

def tokenize(s):
    return set(re.findall(r'\w+', s.lower()))

# tokenize("Machine learning is fun") → {"machine", "learning", "is", "fun"}


# Map each word to the list of strings (or indices) that contain it.
from collections import defaultdict

def build_inverted_index(df_text):
    word_to_strings = defaultdict(set)
    for row in df_text.iterrows():
        row_string = f'{row.label} {row.text}'
        for word in tokenize(row_string):
            word_to_strings[word].add(row)
    return word_to_strings

# Use the inverted index to build clusters: group strings that share a word. This can be done using union-find (disjoint-set) to ensure that indirect connections are also merged.
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

def cluster_strings(strings):
    n = len(strings)
    uf = UnionFind(n)
    inverted_index = build_inverted_index(strings)

    for indices in inverted_index.values():
        indices = list(indices)
        for i in range(1, len(indices)):
            uf.union(indices[0], indices[i])

    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    return clusters
# Once you have the clusters (sets of indices), find the shared token(s) for each cluster:
def shared_token_for_cluster(strings, cluster_indices):
    token_sets = [tokenize(strings[i]) for i in cluster_indices]
    common_tokens = set.intersection(*token_sets)
    return list(common_tokens)
