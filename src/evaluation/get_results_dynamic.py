import pandas as pd
from src.dynamic_approach.optimized_search import the_function

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

# Choose the method
method = "hdbscan"

# Execute the queries
for index, row in queries.iterrows():
    # Get the top 100 results
    results = the_function(row["query"], k=100, method=method)

    # Load the texts of the dataset
    text = pd.read_pickle("wikipedia-text-data-no-disambiguation.pkl.gzip", compression="gzip")
    
    # Add the texts to the retrieved documents
    df = pd.DataFrame(results).set_index("doc_id")
    merged_df = df.merge(text, left_on="doc_id", right_on="d_id", how="left")
    merged_df["q_id"] = row["id"]

    # Add the results to the total results
    total_results = pd.concat([total_results, merged_df[:100]])

# Save the results
total_results.to_csv("dynamic_results.csv")
