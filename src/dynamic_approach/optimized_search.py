import os
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import gc
import time
from sentence_transformers import SentenceTransformer

try:
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    CUML_AVAILABLE = True
except ImportError:
    print("Warning: CuML not available, falling back to CPU clustering")
    from sklearn.cluster import KMeans, HDBSCAN
    CUML_AVAILABLE = False

class OptimizedDocumentClusterer:
    def __init__(self, embeddings_path: str, text_path: str, model_name: str = "all-mpnet-base-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        
        self.embeddings_path = embeddings_path
        self.text_path = text_path
        
        print("Loading text data...")
        self.text_df = pd.read_pickle(text_path, compression='gzip')
        print(f"Loaded {len(self.text_df)} text documents")
        
        # Pre-process embeddings for faster search
        self._prepare_embeddings()
        # Create doc_id mapping to handle format differences
        self._create_doc_id_mapping()
    
    def _prepare_embeddings(self):
        """Pre-process embeddings into manageable chunks with caching"""
        print("Preparing embeddings for search...")
        
        # Check if we have a preprocessed cache
        cache_path = self.embeddings_path.replace('.pkl.gzip', '_cache.npz')
        
        if os.path.exists(cache_path):
            print("Loading cached embeddings...")
            cache = np.load(cache_path, allow_pickle=True)
            self.embeddings_array = cache['embeddings']
            self.doc_ids = cache['doc_ids'].tolist()
            print(f"Loaded cached {len(self.embeddings_array)} embeddings")
        else:
            print("Creating embeddings cache...")
            # Load and convert embeddings
            embeddings_df = pd.read_pickle(self.embeddings_path, compression='gzip')
            
            print("Converting embeddings to numpy arrays...")
            
            # CRITICAL FIX: Handle embeddings DataFrame with wrong index format
            
            # Debug: Check what the original embeddings DataFrame index looks like
            print(f"DEBUG - Original embeddings DataFrame:")
            print(f"  Shape: {embeddings_df.shape}")
            print(f"  Index type: {type(embeddings_df.index)}")
            print(f"  First 5 index values: {embeddings_df.index[:5].tolist()}")
            print(f"  Index dtype: {embeddings_df.index.dtype}")
            print(f"  Columns: {embeddings_df.columns.tolist()}")
            
            # Check if embeddings DataFrame has a 'doc_id' column (fallback)
            if 'doc_id' in embeddings_df.columns:
                print("Found 'doc_id' column, using that for doc_ids")
                original_doc_ids = embeddings_df['doc_id'].tolist()
            elif embeddings_df.index.dtype == 'object' and isinstance(embeddings_df.index[0], str):
                print("Index appears to be string-based doc_ids, using index")
                original_doc_ids = embeddings_df.index.tolist()
            else:
                print("WARNING: Embeddings index appears to be numeric, attempting to reconstruct doc_ids")
                # Reconstruct doc_ids based on position (this assumes 1-based indexing)
                # This matches the pattern we see in text_df: d_0000001, d_0000002, etc.
                original_doc_ids = [f"d_{i+1:07d}" for i in range(len(embeddings_df))]
                print(f"Reconstructed doc_ids format: {original_doc_ids[:5]}")
            
            # Get embeddings in the same order as the DataFrame
            embeddings_list = embeddings_df['embedding'].tolist()
            self.embeddings_array = np.array(embeddings_list, dtype=np.float32)
            
            # Store corrected doc_ids - now position i in embeddings_array corresponds to doc_ids[i]
            self.doc_ids = original_doc_ids
            
            # Validation: check alignment
            print(f"✅ Position-based alignment created:")
            print(f"   Array length: {len(self.embeddings_array)}")
            print(f"   Doc_ids length: {len(self.doc_ids)}")
            print(f"   First 3 position->doc_id mappings: {[(i, self.doc_ids[i]) for i in range(min(3, len(self.doc_ids)))]}")
            
            # Save cache for faster future loading
            print("Saving embeddings cache...")
            np.savez_compressed(cache_path, 
                            embeddings=self.embeddings_array, 
                            doc_ids=np.array(self.doc_ids, dtype=object))
            
            # Clean up
            del embeddings_df, embeddings_df_reset, embeddings_list
            gc.collect()
            
        print(f"Prepared {len(self.embeddings_array)} embeddings of dimension {self.embeddings_array.shape[1]}")
        print(f"Memory usage: {self.embeddings_array.nbytes / (1024**3):.2f} GB")
        print(f"Sample doc IDs: {self.doc_ids[:5]}")  # Debug info
        
    
    def _create_doc_id_mapping(self):
        """Create mapping between embedding doc_ids and text doc_ids"""
        print("Creating doc_id mapping...")
        
        # Analyze the formats
        sample_embedding_ids = self.doc_ids[:5]
        sample_text_ids = self.text_df.index[:5].tolist()
        
        print(f"Embedding doc_ids format: {sample_embedding_ids}")
        print(f"Text doc_ids format: {sample_text_ids}")
        
        # Check if they match directly
        if set(sample_embedding_ids).intersection(set(sample_text_ids)):
            print("Doc_ids match directly, no mapping needed")
            self.doc_id_mapping = None
            return
        
        # Create mapping based on pattern analysis
        self.doc_id_mapping = {}
        self.reverse_mapping = {}
        
        # Try to detect the pattern
        if all(isinstance(text_id, str) and text_id.startswith('d_') for text_id in sample_text_ids):
            # Pattern: embedding_id -> 'd_' + zero-padded(embedding_id + 1)
            print("Detected pattern: numeric -> 'd_XXXXXXX' format")
            
            for embedding_id in self.doc_ids:
                try:
                    # Convert to 1-based index and format as text_id
                    text_id = f"d_{embedding_id + 1:07d}"
                    
                    # Only add mapping if the text_id exists in text data
                    if text_id in self.text_df.index:
                        self.doc_id_mapping[embedding_id] = text_id
                        self.reverse_mapping[text_id] = embedding_id
                except:
                    continue
            
            print(f"Created mapping for {len(self.doc_id_mapping)} documents")
            print(f"Sample mappings: {dict(list(self.doc_id_mapping.items())[:3])}")
            
        else:
            print("Could not detect doc_id pattern, will attempt direct mapping")
            self.doc_id_mapping = None
    
    def _map_to_text_id(self, embedding_doc_id):
        """Convert embedding doc_id to text doc_id"""
        if self.doc_id_mapping is None:
            return embedding_doc_id
        return self.doc_id_mapping.get(embedding_doc_id, embedding_doc_id)
    
        print(f"Prepared {len(self.embeddings_array)} embeddings of dimension {self.embeddings_array.shape[1]}")
        print(f"Memory usage: {self.embeddings_array.nbytes / (1024**3):.2f} GB")
        print(f"Sample doc IDs: {self.doc_ids[:5]}")  # Debug info
    
    def _create_doc_id_mapping(self):
        """Create mapping between embedding doc_ids and text doc_ids"""
        print("Creating doc_id mapping...")
        
        # Analyze the formats
        sample_embedding_ids = self.doc_ids[:5]
        sample_text_ids = self.text_df.index[:5].tolist()
        
        print(f"Embedding doc_ids format: {sample_embedding_ids}")
        print(f"Text doc_ids format: {sample_text_ids}")
        
        # Check if they match directly
        if set(sample_embedding_ids).intersection(set(sample_text_ids)):
            print("Doc_ids match directly, no mapping needed")
            self.doc_id_mapping = None
            return
        
        # Create mapping based on pattern analysis
        self.doc_id_mapping = {}
        self.reverse_mapping = {}
        
        # Try to detect the pattern
        if all(isinstance(text_id, str) and text_id.startswith('d_') for text_id in sample_text_ids):
            # Pattern: embedding_id -> 'd_' + zero-padded(embedding_id + 1)
            print("Detected pattern: numeric -> 'd_XXXXXXX' format")
            
            for embedding_id in self.doc_ids:
                try:
                    # Convert to 1-based index and format as text_id
                    text_id = f"d_{embedding_id + 1:07d}"
                    
                    # Only add mapping if the text_id exists in text data
                    if text_id in self.text_df.index:
                        self.doc_id_mapping[embedding_id] = text_id
                        self.reverse_mapping[text_id] = embedding_id
                except:
                    continue
            
            print(f"Created mapping for {len(self.doc_id_mapping)} documents")
            print(f"Sample mappings: {dict(list(self.doc_id_mapping.items())[:3])}")
            
        else:
            print("Could not detect doc_id pattern, will attempt direct mapping")
            self.doc_id_mapping = None
    
    def _map_to_text_id(self, embedding_doc_id):
        """Convert embedding doc_id to text doc_id"""
        if self.doc_id_mapping is None:
            return embedding_doc_id
        return self.doc_id_mapping.get(embedding_doc_id, embedding_doc_id)
    
    def _map_from_text_id(self, text_doc_id):
        """Convert text doc_id to embedding doc_id"""
        if self.doc_id_mapping is None:
            return text_doc_id
        return self.reverse_mapping.get(text_doc_id, text_doc_id)
    
    def _chunked_similarity_search(self, query_embedding: np.ndarray, k: int = 1000, chunk_size: int = 100000) -> Tuple[List, List]:
        """Optimized chunked similarity search"""
        print(f"Searching through {len(self.embeddings_array)} embeddings...")
        start_time = time.time()
        
        # Use a min-heap approach for memory efficiency
        from heapq import nlargest
        
        all_results = []
        
        # Process in chunks
        for i in range(0, len(self.embeddings_array), chunk_size):
            chunk_end = min(i + chunk_size, len(self.embeddings_array))
            chunk_embeddings = self.embeddings_array[i:chunk_end]
            
            # Compute similarities for this chunk using optimized numpy
            chunk_scores = np.dot(chunk_embeddings, query_embedding)
            
            # Store results with global indices
            for j, score in enumerate(chunk_scores):
                all_results.append((float(score), i + j))
            
            if i // chunk_size % 10 == 0:  # Progress update every 10 chunks
                print(f"Processed {i//chunk_size + 1}/{(len(self.embeddings_array)-1)//chunk_size + 1} chunks")
        
        # Get top k results efficiently
        top_results = nlargest(k, all_results)
        top_scores, top_indices = zip(*top_results)
        
        # Convert to doc IDs
        top_doc_ids = [self.doc_ids[idx] for idx in top_indices]
        
        search_time = time.time() - start_time
        print(f"Search completed in {search_time:.2f} seconds")
        
        return list(top_doc_ids), list(top_scores)
    
    def _gpu_similarity_search(self, query_embedding: np.ndarray, k: int = 1000, batch_size: int = 50000) -> Tuple[List, List]:
        """GPU-accelerated similarity search with proper doc_id preservation"""
        print("Using GPU-accelerated search...")
        start_time = time.time()
        
        try:
            query_gpu = torch.from_numpy(query_embedding).to(self.device)
            
            # Store all results with proper indices
            all_results = []  # List of (score, global_index) tuples
            
            # Process in batches to avoid GPU memory issues
            for i in range(0, len(self.embeddings_array), batch_size):
                batch_end = min(i + batch_size, len(self.embeddings_array))
                batch_embeddings = torch.from_numpy(self.embeddings_array[i:batch_end]).to(self.device)
                
                # Compute similarities for this batch
                batch_similarities = torch.matmul(batch_embeddings, query_gpu)
                
                # Convert to CPU and store with global indices
                batch_scores = batch_similarities.cpu().numpy()
                
                # Add results with correct global indices
                for j, score in enumerate(batch_scores):
                    global_idx = i + j
                    all_results.append((float(score), global_idx))
                
                # Clean up GPU memory for this batch
                del batch_embeddings, batch_similarities
                
                if i // batch_size % 5 == 0:
                    print(f"GPU batch {i//batch_size + 1}/{(len(self.embeddings_array)-1)//batch_size + 1}")
            
            # Sort all results by score (descending) and take top k
            all_results.sort(key=lambda x: x[0], reverse=True)
            top_k_results = all_results[:k]
            
            # Extract scores and indices
            final_scores, final_indices = zip(*top_k_results)
            
            # Convert to doc IDs using the correct indices
            top_doc_ids = [self.doc_ids[idx] for idx in final_indices]
            
            # Clean up GPU memory
            del query_gpu
            torch.cuda.empty_cache()
            
            search_time = time.time() - start_time
            print(f"GPU search completed in {search_time:.2f} seconds")
            print(f"Sample results: indices {list(final_indices)[:3]} -> doc_ids {top_doc_ids[:3]}")
            
            return list(top_doc_ids), list(final_scores)
            
        except Exception as e:
            print(f"GPU search failed: {e}")
            print("Falling back to chunked CPU search...")
            return self._chunked_similarity_search(query_embedding, k)
    
    def _cluster_documents(self, query_embedding: np.ndarray, doc_ids: List, scores: List, 
                          method: str = "kmeans", num_clusters: int = 5) -> List[Dict]:
        """Cluster the top documents and select best from each cluster"""
        print(f"Clustering {len(doc_ids)} documents using {method}...")
        
        if len(doc_ids) < num_clusters:
            num_clusters = len(doc_ids)
        
        # Get embeddings for selected documents - improved mapping
        doc_indices = []
        valid_doc_ids = []
        valid_scores = []
        
        for i, doc_id in enumerate(doc_ids):
            try:
                idx = self.doc_ids.index(doc_id)
                doc_indices.append(idx)
                valid_doc_ids.append(doc_id)
                valid_scores.append(scores[i])
            except ValueError:
                print(f"Warning: doc_id {doc_id} not found in embeddings")
                continue
        
        if len(doc_indices) == 0:
            print("No valid document indices found!")
            return []
        
        selected_embeddings = self.embeddings_array[doc_indices]
        print(f"Using {len(selected_embeddings)} valid documents for clustering")
        
        # Perform clustering
        try:
            if method == "kmeans":
                labels = self._kmeans_clustering(selected_embeddings, num_clusters)
            elif method == "hdbscan":
                labels = self._hdbscan_clustering(selected_embeddings)
            else:
                # No clustering, just return top documents
                return [
                    {
                        "doc_id": valid_doc_ids[i],
                        "init_ranking": i + 1,
                        "new_ranking": i + 1,
                        "cluster": "N/A",
                        "similarity_score": valid_scores[i]
                    }
                    for i in range(min(num_clusters, len(valid_doc_ids)))
                ]
            
            # Select best document from each cluster
            cluster_results = self._select_from_clusters(
                labels, selected_embeddings, valid_doc_ids, valid_scores, query_embedding
            )
            
            return cluster_results
            
        except Exception as e:
            print(f"Clustering failed: {e}")
            # Fallback to simple ranking
            return [
                {
                    "doc_id": valid_doc_ids[i],
                    "init_ranking": i + 1,
                    "new_ranking": i + 1,
                    "cluster": f"fallback_{i}",
                    "similarity_score": valid_scores[i]
                }
                for i in range(min(num_clusters, len(valid_doc_ids)))
            ]
    
    def _kmeans_clustering(self, embeddings: np.ndarray, num_clusters: int) -> np.ndarray:
        """Perform K-means clustering"""
        if CUML_AVAILABLE and torch.cuda.is_available():
            # GPU clustering
            embeddings_gpu = cp.asarray(embeddings)
            kmeans = cuKMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_gpu)
            return cp.asnumpy(labels)
        else:
            # CPU clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(embeddings)
    
    def _hdbscan_clustering(self, embeddings: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
        """Perform HDBSCAN clustering"""
        if CUML_AVAILABLE and torch.cuda.is_available():
            # GPU clustering
            embeddings_gpu = cp.asarray(embeddings)
            clusterer = cuHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                cluster_selection_epsilon=0.2
            )
            clusterer.fit(embeddings_gpu)
            return cp.asnumpy(clusterer.labels_)
        else:
            # CPU clustering
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                cluster_selection_epsilon=0.2
            )
            return clusterer.fit_predict(embeddings)
    
    def _select_from_clusters(self, labels: np.ndarray, embeddings: np.ndarray, 
                            doc_ids: List, scores: List, query_embedding: np.ndarray) -> List[Dict]:
        """Select best document from each cluster"""
        unique_clusters = np.unique(labels)
        cluster_results = []
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # HDBSCAN noise
                continue
                
            # Get documents in this cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Find best document in cluster (highest similarity to query)
            cluster_embeddings = embeddings[cluster_indices]
            cluster_similarities = np.dot(cluster_embeddings, query_embedding)
            best_idx_in_cluster = cluster_indices[np.argmax(cluster_similarities)]
            
            cluster_results.append({
                "cluster_id": int(cluster_id),
                "best_idx": int(best_idx_in_cluster),
                "doc_id": doc_ids[best_idx_in_cluster],
                "similarity_score": scores[best_idx_in_cluster],
                "cluster_size": len(cluster_indices)
            })
        
        # Handle HDBSCAN noise points
        noise_mask = labels == -1
        if np.any(noise_mask):
            noise_indices = np.where(noise_mask)[0]
            for idx in noise_indices[:3]:  # Add top 3 noise points
                cluster_results.append({
                    "cluster_id": "noise",
                    "best_idx": int(idx),
                    "doc_id": doc_ids[idx],
                    "similarity_score": scores[idx],
                    "cluster_size": 1
                })
        
        # Sort by similarity score and assign new rankings
        cluster_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        final_results = []
        for rank, result in enumerate(cluster_results, 1):
            final_results.append({
                "doc_id": result["doc_id"],
                "init_ranking": result["best_idx"] + 1,
                "new_ranking": rank,
                "cluster": result["cluster_id"],
                "similarity_score": result["similarity_score"],
                "cluster_size": result["cluster_size"]
            })
        
        return final_results
    
    def search_and_cluster(self, query: str, k: int = 5, method: str = "kmeans", 
                          retrieval_k: int = 1000) -> pd.DataFrame:
        """Main search and clustering function"""
        try:
            print(f"Searching for: '{query}' (method: {method}, k: {k})")
            start_time = time.time()
            
            # Encode query
            query_embedding = self.model.encode(query, convert_to_tensor=False, normalize_embeddings=True)
            query_embedding = query_embedding.astype(np.float32)
            
            # Step 1: Retrieve top candidates
            embedding_size_gb = self.embeddings_array.nbytes / (1024**3)
            
            if torch.cuda.is_available() and embedding_size_gb < 8:  # If reasonable GPU memory usage
                top_doc_ids, top_scores = self._gpu_similarity_search(query_embedding, retrieval_k)
            else:
                top_doc_ids, top_scores = self._chunked_similarity_search(query_embedding, retrieval_k)
            
            if not top_doc_ids:
                print("No documents found")
                return pd.DataFrame()
            
            print(f"Retrieved {len(top_doc_ids)} candidates")
            
            # Step 2: Clustering (if requested)
            if method in ["kmeans", "hdbscan"]:
                # Limit clustering to reasonable number of documents
                cluster_candidates = min(500, len(top_doc_ids))
                cluster_doc_ids = top_doc_ids[:cluster_candidates]
                cluster_scores = top_scores[:cluster_candidates]
                
                results = self._cluster_documents(
                    query_embedding, cluster_doc_ids, cluster_scores, method, k
                )
            else:
                # Simple ranking without clustering
                results = [
                    {
                        "doc_id": top_doc_ids[i],
                        "init_ranking": i + 1,
                        "new_ranking": i + 1,
                        "cluster": "N/A",
                        "similarity_score": top_scores[i]
                    }
                    for i in range(min(k, len(top_scores)))
                ]
            
            if not results:
                print("No results from clustering")
                return pd.DataFrame()
            
            # Step 3: Create DataFrame and join with text data
            print(f"Creating final results for {len(results)} documents...")
            
            # Create results dataframe
            df = pd.DataFrame(results)
            
            # Map embedding doc_ids to text doc_ids for joining
            if self.doc_id_mapping is not None:
                print("Mapping doc_ids for text joining...")
                df['text_doc_id'] = df['doc_id'].apply(self._map_to_text_id)
                
                # Count successful mappings
                mapped_count = df['text_doc_id'].isin(self.text_df.index).sum()
                print(f"Successfully mapped {mapped_count}/{len(df)} doc_ids to text format")
                
                # Set text_doc_id as index for joining
                df = df.set_index('text_doc_id')
                
                # Keep original doc_id as a column
                df['original_doc_id'] = df['doc_id']
            else:
                # No mapping needed, use doc_id directly
                df = df.set_index('doc_id')
            
            # Debug info
            print(f"Results columns: {df.columns.tolist()}")
            print(f"Results index (for joining): {df.index[:3].tolist()}")
            print(f"Text data shape: {self.text_df.shape}")
            print(f"Text data columns: {self.text_df.columns.tolist()}")
            print(f"Sample text data index: {self.text_df.index[:5].tolist()}")
            
            # Join with text data
            merged_df = df.join(self.text_df, how='left')
            
            # Check for missing joins
            missing_text = merged_df[merged_df[self.text_df.columns].isnull().any(axis=1)]
            if len(missing_text) > 0:
                print(f"Warning: {len(missing_text)} documents missing text data")
                missing_ids = missing_text.index[:5].tolist()
                print(f"Sample missing doc_ids: {missing_ids}")
            else:
                print("✅ All documents successfully joined with text data!")
            
            # Sort by new_ranking and limit to k results
            merged_df = merged_df.sort_values('new_ranking').head(k)
            
            total_time = time.time() - start_time
            print(f"Total search and clustering time: {total_time:.2f} seconds")
            print(f"Final results shape: {merged_df.shape}")
            print(f"Final columns: {merged_df.columns.tolist()}")
            
            return merged_df
            
        except Exception as e:
            print(f"Error in search_and_cluster: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def verify_search_integrity(self, query: str, k: int = 10) -> Dict:
        """Verify that search results preserve doc_ids correctly"""
        print("=== VERIFYING SEARCH INTEGRITY ===")
        
        query_embedding = self.model.encode(query, convert_to_tensor=False, normalize_embeddings=True)
        query_embedding = query_embedding.astype(np.float32)
        
        # Test both GPU and CPU methods with small k
        embedding_size_gb = self.embeddings_array.nbytes / (1024**3)
        
        results = {}
        
        # Test CPU method
        print("Testing CPU method...")
        cpu_doc_ids, cpu_scores = self._chunked_similarity_search(query_embedding, k)
        results['cpu'] = {
            'doc_ids': cpu_doc_ids,
            'scores': cpu_scores,
            'valid_doc_ids': all(doc_id in self.doc_ids for doc_id in cpu_doc_ids),
            'unique_doc_ids': len(set(cpu_doc_ids)) == len(cpu_doc_ids)
        }
        
        # Test GPU method if available
        if torch.cuda.is_available() and embedding_size_gb < 8:
            print("Testing GPU method...")
            gpu_doc_ids, gpu_scores = self._gpu_similarity_search(query_embedding, k)
            results['gpu'] = {
                'doc_ids': gpu_doc_ids,
                'scores': gpu_scores,
                'valid_doc_ids': all(doc_id in self.doc_ids for doc_id in gpu_doc_ids),
                'unique_doc_ids': len(set(gpu_doc_ids)) == len(gpu_doc_ids)
            }
            
            # Compare CPU vs GPU results
            if len(cpu_doc_ids) > 0 and len(gpu_doc_ids) > 0:
                # Check if top results are similar (they should be identical or very close)
                top_3_cpu = set(cpu_doc_ids[:3])
                top_3_gpu = set(gpu_doc_ids[:3])
                overlap = len(top_3_cpu.intersection(top_3_gpu))
                results['cpu_gpu_consistency'] = {
                    'top_3_overlap': overlap,
                    'score_diff': abs(cpu_scores[0] - gpu_scores[0]) if cpu_scores and gpu_scores else 0
                }
        
        # Print detailed results
        for method, data in results.items():
            if method == 'cpu_gpu_consistency':
                print(f"\nCPU vs GPU consistency:")
                print(f"  Top 3 overlap: {data['top_3_overlap']}/3")
                print(f"  Top score difference: {data['score_diff']:.6f}")
            else:
                print(f"\n{method.upper()} method:")
                print(f"  Valid doc_ids: {data['valid_doc_ids']}")
                print(f"  Unique doc_ids: {data['unique_doc_ids']}")
                print(f"  Sample doc_ids: {data['doc_ids'][:3]}")
                print(f"  Sample scores: {[f'{s:.4f}' for s in data['scores'][:3]]}")
                
                # Check for potential issues
                if not data['valid_doc_ids']:
                    invalid_ids = [doc_id for doc_id in data['doc_ids'] if doc_id not in self.doc_ids]
                    print(f"  WARNING: Invalid doc_ids found: {invalid_ids[:5]}")
                
                if not data['unique_doc_ids']:
                    print(f"  WARNING: Duplicate doc_ids found!")
        
        print("=== END INTEGRITY CHECK ===")
        return results
        """Inspect the data structure for debugging"""
        print("=== DATA INSPECTION ===")
        print(f"Embeddings: {len(self.embeddings_array)} documents")
        print(f"Text data: {len(self.text_df)} documents")
        print(f"Embeddings doc_ids type: {type(self.doc_ids[0]) if self.doc_ids else 'None'}")
        print(f"Text data index type: {type(self.text_df.index[0]) if len(self.text_df) > 0 else 'None'}")
        print(f"Sample embedding doc_ids: {self.doc_ids[:5]}")
        print(f"Sample text doc_ids: {self.text_df.index[:5].tolist()}")
        
        # Check for overlap
        embedding_ids_set = set(self.doc_ids)
        text_ids_set = set(self.text_df.index)
        overlap = embedding_ids_set.intersection(text_ids_set)
        print(f"Overlapping doc_ids: {len(overlap)} out of {len(embedding_ids_set)} embeddings")
        
        if len(overlap) < len(embedding_ids_set):
            missing_in_text = embedding_ids_set - text_ids_set
            print(f"Doc_ids in embeddings but not in text: {len(missing_in_text)}")
            print(f"Sample missing: {list(missing_in_text)[:5]}")
        
        print(f"Text data columns: {self.text_df.columns.tolist()}")
        print("=== END INSPECTION ===")
        
        return {
            "embeddings_count": len(self.embeddings_array),
            "text_count": len(self.text_df),
            "overlap_count": len(overlap),
            "text_columns": self.text_df.columns.tolist()
        }
        """Get current memory usage statistics"""
        embedding_size = self.embeddings_array.nbytes / (1024**3)
        
        gpu_info = "N/A"
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            cached = torch.cuda.memory_reserved() / (1024**3)
            gpu_info = f"Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
        
        return {
            "embeddings_size_gb": embedding_size,
            "num_documents": len(self.embeddings_array),
            "gpu_memory": gpu_info
        }
    
    def get_memory_usage(self):
        """Get current memory usage statistics"""
        embedding_size = self.embeddings_array.nbytes / (1024**3)

        gpu_info = "N/A"
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            cached = torch.cuda.memory_reserved() / (1024**3)
            gpu_info = f"Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"

        return {
            "embeddings_size_gb": embedding_size,
            "num_documents": len(self.embeddings_array),
            "gpu_memory": gpu_info
        }
        
    def clear_memory(self):
        """Clean up memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# Enhanced usage function with all your original functionality
def the_function(query: str, k: int = 5, method: str = "hdbscan", 
                embeddings_path: str = None, text_path: str = None, debug: bool = False):
    """Enhanced version of your original the_function"""
    
    # Use your original paths if not provided
    if embeddings_path is None or text_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_path = os.path.abspath(os.path.join(script_dir, "../.."))
        text_path = os.path.join(repo_path, "data/wikipedia/wikipedia-text-data-no-disambiguation.pkl.gzip")
        embeddings_path = os.path.join(repo_path, "data/wikipedia/combined_embeddings.pkl.gzip")
    
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # repo_path = os.path.abspath(os.path.join(script_dir, "../.."))
        # text_path = os.path.join(repo_path, "data/wikipedia/testdata/raw/jaguar.pkl.gzip")
        # embeddings_path = os.path.join(repo_path, "data/wikipedia/testdata/embedded/jaguar_embeddings.pkl.gzip")
        
    try:
        clusterer = OptimizedDocumentClusterer(embeddings_path, text_path)
        
        # # Debug data structure if requested
        # if debug:
        #     data_info = clusterer.inspect_data()
        #     print(f"Data inspection: {data_info}")
        
        # Print memory usage
        memory_info = clusterer.get_memory_usage()
        print(f"Memory usage: {memory_info}")
        
        # Perform search and clustering
        results = clusterer.search_and_cluster(query, k=k, method=method, retrieval_k=1000)
        
        # Verify results contain expected data
        if not results.empty:
            print(f"Results shape: {results.shape}")
            print(f"Results columns: {results.columns.tolist()}")
            
            # Check if key columns exist
            expected_cols = ['new_ranking', 'similarity_score', 'cluster']
            missing_cols = [col for col in expected_cols if col not in results.columns]
            if missing_cols:
                print(f"Warning: Missing expected columns: {missing_cols}")
            
            # Check if text data was joined successfully
            text_cols = [col for col in results.columns if col not in expected_cols + ['init_ranking']]
            if text_cols:
                print(f"Text columns joined: {text_cols}")
            else:
                print("Warning: No text columns found in results")
        
        # Clean up
        clusterer.clear_memory()
        
        return results
        
    except Exception as e:
        print(f"the_function failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

if __name__ == "__main__":
    # Simple test
    results = the_function("jaguar animal", k=5, method="hdbscan")
    print("\nResults:")
    print(results.head())