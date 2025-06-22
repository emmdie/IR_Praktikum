# Queries and Evaluation tools

This folder contains three files used for our IR system that works with homonyms.

All files have been updated so they match the correct document IDs from the Wikipedia dataset.

## Files

### 1. `corpus.jsonl`

- This file has the homonymns with document names
- Each line is a document, with a title and an ID.
- The IDs have been fixed to match Marias Wikipedia article IDs.

### 2. `qrels.tsv`

- This file says which documents are relevant for which queries.
- Each line has a query ID, a document ID, and a number showing how relevant it is.
- The document IDs also now should match the correct Wikipedia ones.

### 3. `queries.jsonl`

- This file has the search queries.
- Each line has a query ID and the search text.

## Why This Matters

Before, the files didn’t match up — some document IDs pointed to the wrong articles.  
Now, everything uses the same correct IDs from Wikipedia, so the system can work properly. 
If code is needed I can provide it. 
