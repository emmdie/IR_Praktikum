import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import matplotlib.pyplot as plt

# Load the results
df_results = pd.read_csv("dynamic_results.csv") # Insert the file with results for the respective approach here
df_results.rename(columns={"Unnamed: 0": "position"}, inplace="True")

# Load the ground truth
df_ground_truth = pd.read_csv("qrels.tsv", sep="\t", header=None, names=["query_id", "document_id", "score"])

# Get the query ids
retrieved = df_results.groupby("q_id")["d_id"].apply(set)
relevant = df_ground_truth.groupby("query_id")["document_id"].apply(set)

all_queries = retrieved.index

precision_recall = []

# Load SBERT
model = SentenceTransformer('all-mpnet-base-v2', device="cuda")

for query_id in all_queries:
  # Get the documents for this query from the results and the ground truth
  retrieved_docs = retrieved.get(query_id, set())
  relevant_docs = relevant.get(query_id, set())

  # Get the texts of the retrieved documents
  retrieved_texts = df_results[
      (df_results["q_id"] == query_id) &
      (df_results["d_id"].isin(retrieved_docs))
  ]["text"].tolist()

  # Convert the retrieved texts to strings if they aren't already
  retrieved_texts = [str(text) for text in retrieved_texts if isinstance(text, (str, int, float)) or text is None]

  if not retrieved_texts:
      # Handle the case where there are no retrieved texts for this query
      precision_recall.append({
          "query_id": query_id,
          "precision": 0,
          "recall": 0,
          "heterogenity_score": 0
      })
      continue

  # Calculate the cosine similarities between the texts
  embeddings_tensor = model.encode(retrieved_texts, convert_to_tensor=True, normalize_embeddings=True)
  similarity_matrix = util.cos_sim(embeddings_tensor, embeddings_tensor)

  # Calculate the total similarity between the retrieved document texts
  upper_triangle = torch.triu(similarity_matrix, diagonal = 1)
  total_similarity = upper_triangle.sum().item()

  # Calculate the heterogenity score
  expected_sum = sum(range(1, len(retrieved_texts) + 1))
  heterogenity_score = 1.0 - total_similarity / expected_sum if expected_sum > 0 else 0

  # Calculate precision and recall
  true_positives = len(retrieved_docs & relevant_docs)
  precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
  recall = true_positives / len(relevant_docs) if relevant_docs else 0

  # Append the calculated metrics the total results
  precision_recall.append({
      "query_id": query_id,
      "precision": precision,
      "recall": recall,
      "heterogenity_score": heterogenity_score
  })

df_metrics = pd.DataFrame(precision_recall)

def compute_f1(row):
  p = row["precision"]
  r = row["recall"]
  if p + r == 0:
    return 0
  return 2 * (p * r) / (p + r)

# Add the F1 measure to the metrics
df_metrics["f1"] = df_metrics.apply(compute_f1, axis=1)

def compute_f1_heterogenous(row):
  p = row["precision"]
  h = row["heterogenity_score"]
  if p + h == 0:
    return 0
  return 2 * (p * h) / (p + h)

# Add the F1 heterogenity measure to the metrics
df_metrics["f1_heterogenous"] = df_metrics.apply(compute_f1_heterogenous, axis=1)

# Create a plot and save it
columns_to_plot = ["precision", "recall", "heterogenity_score", "f1", "f1_heterogenous"]
df_metrics[columns_to_plot].boxplot()
plt.ylim(0, 1)
plt.savefig("evaluation_visualization.png")