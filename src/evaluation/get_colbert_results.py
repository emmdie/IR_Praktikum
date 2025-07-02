from ragatouille import RAGPretrainedModel
import pandas as pd

# Load the dataset
df = pd.read_pickle("small_dataset.pkl.gzip", compression='gzip')

# Load the queries
with open("queries.txt", 'r', encoding='utf-8') as f:
    queries = [line.strip() for line in f if line.strip()]

# Create a dataframe to later save the results
total_results = pd.DataFrame(columns=["rank", "score", "d_id", "q_id", "text", "label"])

# Load Colbert
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Encode the documents
RAG.encode([row["text"] for index, row in df.iterrows()], document_metadatas=[{"d_id": row["d_id"], "label": row["label"], "text": row["text"]} for index, row in df.iterrows()])

# Execute the queries
for query in queries:
    results = RAG.search_encoded_docs(query=query, k=100)

    df_results = pd.DataFrame(results)

    df_results["d_id"] = df_results["document_metadata"].apply(lambda x: x.get("d_id") if isinstance(x, dict) else None)

    df_results["label"] = df_results["document_metadata"].apply(lambda x: x.get("label") if isinstance(x, dict) else None)

    df_results["text"] = df_results["document_metadata"].apply(lambda x: x.get("text") if isinstance(x, dict) else None)
    df_results["query"] = query
    df_results.drop(columns=["document_metadata", "content", "result_index"], inplace=True)

    # Append the results of this query to the total results
    total_results = pd.concat([total_results, df_results])

# Save the results from Colbert
total_results.to_csv("colbert_results.csv")