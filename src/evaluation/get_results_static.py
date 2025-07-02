import pandas as pd
from src.static_approach.SBERT_static_search import sbert_static_search

# Load the queries 
queries = pd.read_json("data/queries-and-eval/queries.jsonl", lines=True)
"""
Version for small testset:

# Load the queries
with open("queries.txt", 'r', encoding='utf-8') as f:
    queries = [line.strip() for line in f if line.strip()]

"""

# Create a dataframe to store the results in
total_results = pd.DataFrame(columns=["init_ranking", "new_ranking", "cluster", "similarity_score", "d_id", "label", "text", "q_id"])

# Execute the queries
for index, row in queries.iterrows():
    # Get the top 100 results
    results = sbert_static_search(row["query"], num_docs_to_retrieve=100)

    # Load the texts of the dataset
    text = pd.read_pickle("wikipedia-text-data-no-disambiguation.pkl.gzip", compression="gzip")
    
    # Add the texts to the retrieved documents
    df = pd.DataFrame(results).set_index("doc_id")
    merged_df = df.merge(text, left_on="doc_id", right_on="d_id", how="left")
    merged_df["q_id"] = row["id"]

    # Add the results to the total results
    total_results = pd.concat([total_results, merged_df[:100]])

# Save the results
total_results.to_csv("static_results.csv")