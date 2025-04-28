#ToDo wir schauen grad nur das letzte Layer an, vlt würden mehr besser performen

import torch
from transformers import RobertaTokenizer, RobertaModel
import torch.nn.functional as F #Für die Cosine simil
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name) #Nicht Auto tokenizer, der geht nicht unbedingt gut mit Roberta
model =  RobertaModel.from_pretrained(model_name)
model.eval()

def get_embedding(sentence, target_tokens):
    sentence = " " + sentence.lower() # Wir müssen das lowern und ein Leerzeichen davor setzen, damit das Tokenisieren konsistent ist und das Matching funktioniert
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Wir suchen die Indizes der Target Tokens, nicht nur einen wie bei Bert, weil der Roberta Tokenizer das Wort eventuell aufdröselt
    for i in range(len(tokens) - len(target_tokens) + 1):
        if tokens[i:i+len(target_tokens)] == target_tokens:
            token_index = (i, i+len(target_tokens))
            break
    else:
        # Falls das exakte Token nicht gefunden wird, versuchen wir eine "zusammengesetzte" Übereinstimmung
        flat_target = "".join(t.lstrip("Ġ") for t in target_tokens)
        for i in range(len(tokens) - 1):
            merged = "".join(t.lstrip("Ġ") for t in tokens[i:i+2])
            if merged == flat_target:
                token_index = (i, i+2)
                break
        else:
            raise ValueError(f"Die tokens {target_tokens} wurden nicht in s: {tokens} gefunden")

    with torch.no_grad(): # Wir inferieren nur, keine Backpropagation, deswegen no_grad()
        outputs = model(**inputs) # Wir machen einen Forward-Pass des Dictionaries unserer tokenisierten Satzdaten, wobei ** aus dem Dict Keyword-Arguments für unser Modell macht
    token_embeddings = outputs.last_hidden_state[0] # Gibt einen Tensor zurück mit Form (batch_size, seq_len, hidden_size); bei uns ist batch_size 1, wir schauen also direkt [0] an.

    selected_embeddings = token_embeddings[token_index[0]:token_index[1]] # Wir nehmen die Embeddings der Token-Range
    return selected_embeddings.mean(dim=0) # Mittelwert, um eine einzige Repräsentation für das möglicherweise mehrteilige Token zu kriegen

def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item() #Unsqueze damit input tensor batch dimension hat

    #Generiere die Embeddings für das Homonym in den Kontexten
def process_query_and_documents(query, documents, target_word):
    target_tokens = tokenizer.tokenize(" " + target_word) #Wir brauchen den Space für Roberta, die will das so :(
    query_embedding = get_embedding(query, target_tokens)
    doc_embeddings = []
    for doc in documents:
        doc_embedding = get_embedding(doc, target_tokens)
        doc_embeddings.append(doc_embedding)
    return query_embedding, doc_embeddings

#Wir wollen Dokumente danach sortieren, wie ähnlich ihre embeddings zu den query embeddings sind. Dazu nehmen wir cosine similiarity
def rank_documents(query_embedding, doc_embeddings):
    scores = []
    for doc_embedding in doc_embeddings:
        score = cosine_similarity(query_embedding, doc_embedding)
        scores.append(score)
    return np.argsort(scores)[::-1] #Absteigende Sortierung

def retrieve_documents(query, documents, target_word):
    query_embedding, doc_embeddings = process_query_and_documents(query, documents, target_word)
    ranked_indices = rank_documents(query_embedding, doc_embeddings)
    
    ranked_documents = [documents[i] for i in ranked_indices]
    return ranked_documents

query = "How does the road winds?"
query2 = "The winds blow really really friggin hard"
documents = [
    "The road winds through the valley.",
    "The winds were strong that day.",
    "Winds blow across the desert.",
    "The winding path winds through the forest."
]
target_word = "winds"

ranked_docs = retrieve_documents(query, documents, target_word)
ranked_docs2 = retrieve_documents(query2, documents, target_word)

print("Gerankte Dokumente für die Query :" + str(query))
for doc in ranked_docs:
    print(doc)

print("Gerankte Dokumente für die Query :" +str(query2))
for doc in ranked_docs2:
    print(doc)

