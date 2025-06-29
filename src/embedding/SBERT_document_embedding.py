import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-mpnet-base-v2", device=device)
df = pd.read_pickle("data/wikipedia/split-data-no-disambiguation/wikipedia-text-data-no-disambiguation_0.pkl.gzip", compression="gzip")


documents = df.iloc[:, 2].tolist()
embeddings = model.encode(documents)
ids = df["d_id"].tolist()
labels = df["label"].tolist()
new_df = pd.DataFrame({
    'd_id': ids,
    'label': labels,
    'embedding': list(embeddings)
    })
new_df.set_index('d_id', inplace=True)
new_df.to_pickle("embeddings_0.pkl.gzip", compression="gzip")
