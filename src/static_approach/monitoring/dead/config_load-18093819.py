# General config
HPC_EXECUTION = True
CUML = False # enable GPU features
CLUSTERING_STRATEGY = 'mini_batch_kmeans' # kmeans or hdbscan or mini_batch_kmeans
PCA_ENABLED = False
SAMPLING_FRACTION = 1.0  # set between 0 and 1
SKIP_LARGE_CATEGORIES = False
STOP_WORDS_EXCLUDED = False

# HDB config
CLUSTERING_METRIC = 'euclidean' # euclidean or cosine if using hdbscan - ONLY RELEVANT IF CLUSTERING STRATEGY hdbscan

# Kmeans config
KMEANS_K = 8                # Number of clusters/centroids
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
PATH_TO_REPR = "IR_Praktikum/data"


# FAILED BECAUSE OF SEGFAULT
"""[frsc115a@login1.barnard martin]$ cat slurm-err/18093819.err 
OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata.
To avoid this warning, please rebuild your copy of OpenBLAS with a larger NUM_THREADS setting
or set the environment variable OPENBLAS_NUM_THREADS to 64 or lower
Fatal Python error: Segmentation fault

Thread 0x00007ffb7effd700 (most recent call first):
<no Python frame>

Thread 0x00007ffb7ffff700 (most recent call first):
<no Python frame>

Thread 0x00007ffb4c7d8700 (most recent call first):
<no Python frame>

Thread 0x00007ffb48fd1700 (most recent call first):
<no Python frame>

Thread 0x00007ffb3ffbf700 (most recent call first):
<no Python frame>

...
srun: error: n1504: task 0: Segmentation fault
"""
