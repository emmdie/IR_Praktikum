# Capella
## 464939, 464954
Config
SAMPLING_FRACTION = 1.0  # set between 0 and 1
SKIP_LARGE_CATEGORIES = False
STOP_WORDS_EXCLUDED = False
PCA_ENABLED = True
CLUSTERING_STRATEGY = 'hdbscan' # kmeans or hdbscan or mini_batch_kmeans
CLUSTERING_METRIC = 'euclidean' # euclidean or cosine if using hdbscan - ONLY RELEVANT IF CLUSTERING STRATEGY hdbscan
KMEANS_K = 8 # ONLY RELEVANT IF CLUSTERING STRATGY kmeans
CUML = True
HPC_EXECUTION = True

Relevant if SKIP_LARGE_CATEGORIES is True
LARGE_CATEGORY_CONST = 8000 # ONLY RELEVANT IF SKIP_LARGE_CATEGORIES
PCA_VALUE = 200 # number of components or float between 0 and 1 indicating captured variance

path_to_repr_rel = "IR_Praktikum/data/repr_" + CM

os.makedirs(path_to_repr_rel, exist_ok=True)

path_to_doc_data = os.path.join(PWD, "IR_Praktikum/data/wikipedia/split-data-no-disambiguation")
path_to_doc_emb = os.path.join(PWD, "normalize_embeddings")
path_to_representatives = os.path.join(PWD, path_to_repr_rel)

## 465150
Config
SAMPLING_FRACTION = 1.0  # set between 0 and 1
SKIP_LARGE_CATEGORIES = False
STOP_WORDS_EXCLUDED = False
PCA_ENABLED = True
CLUSTERING_STRATEGY = 'hdbscan' # kmeans or hdbscan or mini_batch_kmeans
CLUSTERING_METRIC = 'euclidean' # euclidean or cosine if using hdbscan - ONLY RELEVANT IF CLUSTERING STRATEGY hdbscan
KMEANS_K = 8 # ONLY RELEVANT IF CLUSTERING STRATGY kmeans
CUML = True
HPC_EXECUTION = True

Relevant if SKIP_LARGE_CATEGORIES is True
LARGE_CATEGORY_CONST = 8000 # ONLY RELEVANT IF SKIP_LARGE_CATEGORIES
PCA_VALUE = 200 # number of components or float between 0 and 1 indicating captured variance

path_to_repr_rel = "IR_Praktikum/data/repr_" + CM

os.makedirs(path_to_repr_rel, exist_ok=True)

path_to_doc_data = os.path.join(PWD, "IR_Praktikum/data/wikipedia/split-data-no-disambiguation")
path_to_doc_emb = os.path.join(PWD, "normalize_embeddings")
path_to_representatives = os.path.join(PWD, path_to_repr_rel)

# Barnard
## 18078059 - unexpected keyword batch_size in MiniBatchKMeansClustering

Config
SAMPLING_FRACTION = 1.0  # set between 0 and 1
SKIP_LARGE_CATEGORIES = False
STOP_WORDS_EXCLUDED = False
PCA_ENABLED = True
CLUSTERING_STRATEGY = 'mini_batch_kmeans' # kmeans or hdbscan or mini_batch_kmeans
CLUSTERING_METRIC = 'euclidean' # euclidean or cosine if using hdbscan - ONLY RELEVANT IF CLUSTERING STRATEGY hdbscan
KMEANS_K = 8 # ONLY RELEVANT IF CLUSTERING STRATGY kmeans
CUML = True
HPC_EXECUTION = True

Kmeans config
BATCH_SIZE = 512         # Controls internal processing memory
INIT_SIZE = 2000         # Used for centroid initialization (k-means++)
N_CLUSTERS = 1000        # Number of target clusters
MAX_ITER = 200           # Total iterations to run
REASSIGNMENT_RATIO = 0.01  # Avoid too frequent reinitialization
RANDOM_STATE = 42        # For reproducibility

Relevant if SKIP_LARGE_CATEGORIES is True
LARGE_CATEGORY_CONST = 8000 # ONLY RELEVANT IF SKIP_LARGE_CATEGORIES
PCA_VALUE = 200 # number of components or float between 0 and 1 indicating captured variance

path_to_repr_rel = "IR_Praktikum/data/repr_" + CM

os.makedirs(path_to_repr_rel, exist_ok=True)

path_to_doc_emb = os.path.join(PWD, "normalize_embeddings")
path_to_doc_data = os.path.join(PWD, "IR_Praktikum/data/wikipedia/split-data-no-disambiguation")
path_to_representatives = os.path.join(PWD, path_to_repr_rel)

## 18078072 - unexpected keyword batch_size in MiniBatchKMeansClustering
SAMPLING_FRACTION = 1.0  # set between 0 and 1
SKIP_LARGE_CATEGORIES = False
STOP_WORDS_EXCLUDED = False
PCA_ENABLED = False
CLUSTERING_STRATEGY = 'mini_batch_kmeans' # kmeans or hdbscan or mini_batch_kmeans
CLUSTERING_METRIC = 'euclidean' # euclidean or cosine if using hdbscan - ONLY RELEVANT IF CLUSTERING STRATEGY hdbscan
KMEANS_K = 8 # ONLY RELEVANT IF CLUSTERING STRATGY kmeans
CUML = True
HPC_EXECUTION = True

Kmeans config
BATCH_SIZE = 512         # Controls internal processing memory
INIT_SIZE = 2000         # Used for centroid initialization (k-means++)
N_CLUSTERS = 1000        # Number of target clusters
MAX_ITER = 200           # Total iterations to run
REASSIGNMENT_RATIO = 0.01  # Avoid too frequent reinitialization
RANDOM_STATE = 42        # For reproducibility

Relevant if SKIP_LARGE_CATEGORIES is True
LARGE_CATEGORY_CONST = 8000 # ONLY RELEVANT IF SKIP_LARGE_CATEGORIES
PCA_VALUE = 200 # number of components or float between 0 and 1 indicating captured variance

path_to_repr_rel = "IR_Praktikum/data/repr_" + CM

os.makedirs(path_to_repr_rel, exist_ok=True)

path_to_doc_emb = os.path.join(PWD, "normalize_embeddings")
path_to_doc_data = os.path.join(PWD, "IR_Praktikum/data/wikipedia/split-data-no-disambiguation")
path_to_representatives = os.path.join(PWD, path_to_repr_rel)