import os

def load_documents(*dirs):  # Nimmt beliebig viele Ordner mit txt files
    doc_texts = []
    doc_paths = []
    for dir_path in dirs:
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    doc_texts.append(content)
                    doc_paths.append(file_path)
    return doc_texts, doc_paths