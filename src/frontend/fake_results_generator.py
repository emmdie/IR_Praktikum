import pandas as pd

def generate_fake_results_df() -> pd.DataFrame:
    data = [
        {"doc_id": 182348321, "init_ranking": 55, "category": "Ohrenschmalzentferner", "cluster": 6, "text": "Der Ohrenschmalzentferner schläft nie"},
        {"doc_id": 182348322, "init_ranking": 48, "category": "Kaffeemaschine", "cluster": 2, "text": "Die Kaffeemaschine ist kaputt"},
        {"doc_id": 1823245322, "init_ranking": 13, "category": "Jaguar", "cluster": 2, "text": "Der Jaguar hat den Tofu erlegt"},
        {"doc_id": 1568756722, "init_ranking": 38, "category": "Car", "cluster": 2, "text": "Das Auto hat keinen Tofu erlegt"},
        {"doc_id": 1828857682, "init_ranking": 28, "category": "puma", "cluster": 2, "text": "Eine Schuhmarke, hat nichts mit Jaguaren zu tun"},
        {"doc_id": 1823999992, "init_ranking": 48, "category": "Ksafd", "cluster": 2, "text": "Ein zufälliger String, es scheint keine Naheliegende Verbindung zu "},
    ]
    
    df = pd.DataFrame(data)
    return df


