import pandas as pd
import hashlib
from itertools import combinations

def assign_cluster_colors(df: pd.DataFrame) -> pd.DataFrame:
    def cluster_to_color(cluster_id: int) -> str:
        h = hashlib.md5(str(cluster_id).encode()).hexdigest()
        return f"#{h[:6]}"  # Use first 6 hex chars as color
    
    cluster_color_map = {cid: cluster_to_color(cid) for cid in df['cluster'].unique()}
    df['cluster_color'] = df['cluster'].map(cluster_color_map)
    return df

def top_k_swap_count(df: pd.DataFrame, k: int = 10) -> int:
    """
    Computes the Top-k List Agreement using Swap Count (Kendall Tau distance restricted to top-k).
    df: Ordered by new ranking (row index is ranking).
    init_ranking: Column giving original ranking.
    """
    top_k_df = df.head(k).copy()
    top_k_df["new_rank"] = range(1, len(top_k_df)+1)  # explicit new ranking
    
    doc_to_init_rank = dict(zip(df["doc_id"], df["init_ranking"]))
    
    top_k_df["init_rank"] = top_k_df["doc_id"].map(doc_to_init_rank)

    swap_count = 0
    top_k_docs = top_k_df.to_dict("records")
    
    for doc1, doc2 in combinations(top_k_docs, 2):
        init_order = doc1["init_rank"] - doc2["init_rank"]
        new_order = doc1["new_rank"] - doc2["new_rank"]
        
        if init_order * new_order < 0:
            swap_count += 1

    return swap_count
