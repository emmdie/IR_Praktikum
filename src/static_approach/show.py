import pandas as pd
import matplotlib.pyplot as plt

def highest_doc_freq(index):
    # Create DataFrame from inverted index
    df_freq = pd.DataFrame({
        'term': list(index.keys()),
        'doc_freq': [len(docs) for docs in index.values()]
    })

    # Sort by frequency
    df_freq_sorted = df_freq.sort_values(by='doc_freq', ascending=False).head(100)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(df_freq_sorted['term'], df_freq_sorted['doc_freq'], color='orange')
    plt.xticks(rotation=90)
    plt.title('Top 20 Terms by Document Frequency')
    plt.xlabel('Term')
    plt.ylabel('Number of Documents')
    plt.tight_layout()
    plt.show()

def histogram(value_list, title='Title', x_label='x_label', y_label='y_label'):
    # Step 1: Convert the inverted index to a list of document list lengths
    term_lengths = value_list #[len(docs) for docs in index.values()]

    # Step 2: Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(term_lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def doc_texts_clusterwise(search_results_df : pd.DataFrame):
    search_results = {cluster: data for cluster, data in search_results_df.groupby('cluster')}
    for cluster, retrieved_docs in search_results.items():
        print(f"Cluster {cluster}:")
        for doc_id in retrieved_docs.index:
            # DUPLICATES NOT SHOWN AT THE MOMENT
            # print(search_results_df.loc[doc_id].text)
            print(search_results_df.loc[[doc_id]].iloc[0].text)