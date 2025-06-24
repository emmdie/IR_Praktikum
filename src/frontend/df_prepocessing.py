import pandas as pd
import hashlib

def assign_cluster_colors(df: pd.DataFrame) -> pd.DataFrame:
    def cluster_to_color(cluster_id: int) -> str:
        h = hashlib.md5(str(cluster_id).encode()).hexdigest()
        return f"#{h[:6]}"  # Use first 6 hex chars as color
    
    cluster_color_map = {cid: cluster_to_color(cid) for cid in df['cluster'].unique()}
    df['cluster_color'] = df['cluster'].map(cluster_color_map)
    return df
