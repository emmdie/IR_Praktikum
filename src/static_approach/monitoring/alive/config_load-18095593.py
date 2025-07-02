## THIS IS A CONFIGURATION FILE TO SET PARAMETERS FOR THE LOADING PHASE
# The configuration here is just some configuration that has been used

# General config
HPC_EXECUTION = True
CUML = False # enable GPU features
CLUSTERING_STRATEGY = 'mini_batch_kmeans' # kmeans or hdbscan or mini_batch_kmeans
PCA_ENABLED = False
SAMPLING_FRACTION = 0.1  # set between 0 and 1
SKIP_LARGE_CATEGORIES = False
STOP_WORDS_EXCLUDED = False

# HDB config
CLUSTERING_METRIC = 'euclidean' # euclidean or cosine if using hdbscan - ONLY RELEVANT IF CLUSTERING STRATEGY hdbscan

# Kmeans config
KMEANS_K = 8                # Number of clusters/centroids

# Additional config for Mini Batch Kmeans
BATCH_SIZE = 512            # Controls internal processing memory
INIT_SIZE = 2000            # Used for centroid initialization (k-means++)
N_CLUSTERS = 1000           # Number of target clusters
MAX_ITER = 200              # Total iterations to run
REASSIGNMENT_RATIO = 0.01   # Avoid too frequent reinitialization
RANDOM_STATE = 42           # For reproducibility

# Relevant if SKIP_LARGE_CATEGORIES is True
LARGE_CATEGORY_CONST = 8000 # ONLY RELEVANT IF SKIP_LARGE_CATEGORIES
PCA_VALUE = 200 # number of components or float between 0 and 1 indicating captured variance


PATH_TO_ROOT = "/data/horse/ws/maha780c-IR_Project/martin/" # or os.cwd()
PATH_TO_EMB = "normalize_embeddings"
PATH_TO_DATA = "IR_Praktikum/data/wikipedia/split-data-no-disambiguation"
PATH_TO_REPR = "IR_Praktikum/data/static-approach"
