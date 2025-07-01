# Evaluation on small data set
* Based on files in `data/wikipedia/testdata/raw`
* The respective embeddings files can be found [here](https://datashare.tu-dresden.de/s/XjmdwMznpcP3pje). Use embedded_testdata
* **Don't use files of the category cat**

# Evaluation on all
* `queries.txt` provides a list of queries, where the search term can be found in at least fifty documents
* There is no ground truth - files have to be evaluated against colbert

# Prime Evaluation
* `prime_queries.txt` contains the names of the categories the dataset has been composed of. It is expected that our search engines perform particularily well on these queries
* results_*.txt contain the expected results for each query. Note that these should only be used for **precision** but not recall as thousands of documents are contained in each class.