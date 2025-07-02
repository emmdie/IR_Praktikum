# Capella
## 464955
SAMPLING_FRACTION = 1.0  # set between 0 and 1
SKIP_LARGE_CATEGORIES = False
STOP_WORDS_EXCLUDED = False
PCA_ENABLED = True
CLUSTERING_STRATEGY = 'mini_batch_kmeans' # kmeans or hdbscan or mini_batch_kmeans
CLUSTERING_METRIC = 'cosine' # euclidean or cosine if using hdbscan - ONLY RELEVANT IF CLUSTERING STRATEGY hdbscan
KMEANS_K = 8 # ONLY RELEVANT IF CLUSTERING STRATGY kmeans
CUML = True
HPC_EXECUTION = True

Relevant if SKIP_LARGE_CATEGORIES is True
LARGE_CATEGORY_CONST = 8000 # ONLY RELEVANT IF SKIP_LARGE_CATEGORIES
PCA_VALUE = 0.95 # number of components or float between 0 and 1 indicating captured variance

path_to_repr_rel = "IR_Praktikum/data/repr_small/repr_" + CM

os.makedirs(path_to_repr_rel, exist_ok=True)

path_to_doc_data = os.path.join(PWD, "testdata/raw")
path_to_doc_emb = os.path.join(PWD, "testdata/emb")
path_to_representatives = os.path.join(PWD, path_to_repr_rel)