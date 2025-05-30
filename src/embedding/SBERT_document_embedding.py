import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

print("Hello World!")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-mpnet-base-v2", device=device)
print(model)
df = pd.read_pickle("data/wikipedia/split-data-no-disambiguation/wikipedia-text-data-no-disambiguation_1.pkl.gzip", compression="gzip")
print(df)

documents = df.iloc[:, 2].tolist()
print(documents[:5])
print("Start embedding documents")
embeddings = model.encode(documents)
print("Finished embedding documents")
ids = df["d_id"].tolist()
print(ids[:5])
new_df = pd.DataFrame({'embedding': list(embeddings)})
print(new_df)
new_df.insert(0, 'd_id', ids)
#df = pd.DataFrame({'d_id': ids, 'embedding': embeddings})
print(new_df)
new_df.to_pickle("embeddings_1.pkl.gzip", compression="gzip")