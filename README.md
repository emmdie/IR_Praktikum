## 🚀 Diversifying Search Results for Homonym Queries via Transformer Architectures
This project presents a novel search engine that delivers *diverse* and *relevant* results for homonym queries, leveraging state-of-the-art transformer architectures and semantic clustering. Developed at TU Dresden, it demonstrates advanced skills in NLP, information retrieval, and practical system design.

---

## 📄 Full Report

For a detailed description of the methodology, experiments, and results, see the [full project report (PDF)](IR_Praktikum.pdf).

---

### Table of Contents

- [Project Overview](#project-overview)
- [Research Problem & Motivation](#research-problem--motivation)
- [Technical Approach](#technical-approach)
  - [Static Diversification](#static-diversification)
  - [Dynamic Diversification](#dynamic-diversification)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Evaluation & Results](#evaluation--results)
- [Dataset](#dataset)
- [Team & Credits](#team--credits)

---

## 📝 Project Overview

This repository contains the implementation of a search engine designed to **diversify search results for homonym queries** (e.g., "jaguar" as a car or animal). By combining transformer-based semantic encoding (SBERT) with both static and dynamic diversification strategies, the system ensures users receive results covering multiple meanings of ambiguous queries.

---

## 🎯 Research Problem & Motivation

Traditional search engines often return results focused on a single interpretation of a homonym, failing to address the user's true intent when multiple meanings exist. Our goal was to:

- **Maximize diversity** in search results for single-word, ambiguous queries.
- **Leverage modern NLP techniques** to go beyond keyword matching.
- **Evaluate** both the *relevance* and *semantic diversity* of returned results.

---

## 🛠️ Technical Approach

### Static Diversification

- **Precompute** semantic embeddings for the first paragraph of Wikipedia articles using **SBERT**.
- At query time, **retrieve and rank** results by cosine similarity to the query embedding.
- Ensures fast retrieval with some inherent diversity due to semantic encoding.

### Dynamic Diversification

- At query time, **cluster candidate results** (using scikit-learn) based on their SBERT embeddings.
- **Select top results from different clusters** to maximize semantic diversity.
- Adapts dynamically to the query's ambiguity and the distribution of meanings.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[User Query] --> B[SBERT Embedding]
    B --> C{Diversification Mode}
    C -- Static --> D[Precomputed Embeddings]
    D --> E[Cosine Similarity Ranking]
    C -- Dynamic --> F[Candidate Retrieval]
    F --> G[Clustering (scikit-learn)]
    G --> H[Diverse Result Selection]
    E & H --> I[Front-End Interface]
    I --> J[User]
```

- **Front-End:** User submits query and views results.
- **Back-End:** Handles embedding, retrieval, clustering, and ranking.
- **Evaluation:** Metrics computed for both relevance and diversity.

---

## 🧰 Tech Stack

| Component         | Technology / Library        |
|-------------------|-----------------------------|
| Language          | Python 3.8+                 |
| NLP Models        | SBERT (Sentence-BERT)       |
| ML/Clustering     | K-means and HDBSCAN         |
| Data Processing   | NumPy, Pandas               |
| Front-End         | Textual (UI)                |
| Dataset           | Wikipedia, Wikidata         |

---

## 📊 Evaluation & Results

- **Relevance Metrics:** Precision, Recall, F1 Score
- **Diversity Metrics:** Heterogeneity, F1 Heterogeneity

> **Key Finding:**  
> Both static and dynamic approaches increased the diversity of search results for homonym queries, with dynamic clustering providing the highest heterogeneity without sacrificing relevance.

---

## 📚 Dataset

- **Wikipedia Articles:** Only the *first paragraph* of each article is indexed for concise, relevant information.
- **Wikidata:** Used to validate and enrich semantic diversity.

## 👥 Team & Credits

Developed as part of the "Information Retrieval Praktikum" course at **TU Dresden**.

- Emmanuel Diehl
- Manuel Berger
- Franz Martin Schmidt
- Maria Hampel
- Reiner Frank Stolle

---
