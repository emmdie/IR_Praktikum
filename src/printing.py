from collections import defaultdict



def print_all(doc_classes):
    for doc_class in doc_classes.keys():
        print(f"Document class: {doc_class}")
        for doc in doc_classes[doc_class]:
            print(f"{doc}")

def print_labels(clustering_results, doc_classes, o_type="dict"):
    if o_type == "dict":
        # building dict
        clustering = dict()
        for doc_class in doc_classes:
            clustering[doc_class] = defaultdict(list)
            num_clusters_doc_class = len(clustering_results[doc_class])
            for doc in range(num_clusters_doc_class):
                doc_label = clustering_results[doc_class][doc][0]
                document = doc_classes[doc_class][doc]
                clustering[doc_class][doc_label].append(document)

        # printing dict
        for doc_class in doc_classes:
            print(f"\n\nDocument class: {doc_class}")
            for doc_label in clustering[doc_class]:
                print(f"\n{doc_label}:")
                for doc in clustering[doc_class][doc_label]:
                    print(f"{doc}")
    else:
        for doc_class in doc_classes:
            print(f"{doc_class}")
            num_clusters_doc_class = len(clustering_results[doc_class])
            for doc in range(num_clusters_doc_class):
                doc_label = clustering_results[doc_class][doc][0]
                document = doc_classes[doc_class][doc]
                print(f"{doc_label}: {document}")