#ToDo wir schauen grad nur das letzte Layer an, vlt würden mehr besser performen
#ToDo Cosine auf iwelchen raw ebeddings ist noch schwach
#ToDO Ich muss poolen, weil mein token aufgesplittet wird, das verwäscht das nochmal

import torch
from transformers import RobertaTokenizer, RobertaModel #AutoTokenizer nicht gut für Roberta
import torch.nn.functional as F
import numpy as np

# Load tokenizer and model
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)
model.eval()

def get_embedding(sentence, target_tokens):
    sentence = " " + sentence.lower() #Leerzeichen für Roberta Sonderzeichen, lowercase für matchen
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    #Roberta splittet Tokens, anders als Bert, sind also u.U. mehrere
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
            raise ValueError(f"Tokens {target_tokens} nicht gefunden in {tokens}")

    with torch.no_grad(): # Wir inferieren nur, keine Backpropagation, deswegen no_grad()
        outputs = model(**inputs) # Wir machen einen Forward-Pass des Dictionaries unserer tokenisierten Satzdaten, wobei ** aus dem Dict Keyword-Arguments für unser Modell macht
    token_embeddings = outputs.last_hidden_state[0] # Gibt einen Tensor zurück mit Form (batch_size, seq_len, hidden_size); bei uns ist batch_size 1, wir schauen also direkt [0] an.

    selected_embeddings = token_embeddings[token_index[0]:token_index[1]] # Wir nehmen die Embeddings der Token-Range
    return selected_embeddings.mean(dim=0) # Mittelwert, um eine einzige Repräsentation für das möglicherweise mehrteilige Token zu kriegen

def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

#Generiere die Embeddings für das Homonym in den Kontexten
def process_query_and_documents(query, documents, target_word):
    target_tokens = tokenizer.tokenize(" " + target_word)
    query_embedding = get_embedding(query, target_tokens)
    doc_embeddings = [get_embedding(doc, target_tokens) for doc in documents]
    return query_embedding, doc_embeddings

def retrieve_documents(query, documents, target_word):
    print(f"Ranked documents for query: '{query}'")
    query_embedding, doc_embeddings = process_query_and_documents(query, documents, target_word)

    scores = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]

    ranked_indices = np.argsort(scores)[::-1] #Absteigende Sortierung

    for idx in ranked_indices:
        doc = documents[idx]
        score = scores[idx]
        print(f"{doc} (Cosine Similarity: {score:.4f})")

# Queries and documents
query1 = "The road winds sharply through the rugged landscape, and cars trace its curves like slow-moving shadows pressed against the earth."
query2 = "The winds howl over the plains, flattening grasses and leaving the horizon blurred with flying grit"
documents = [
    "The road winds through the valley.",
    "The winds were strong that day.",
    "Winds blow across the desert.",
    "The winding path winds through the forest."
]
target_word = "winds"

retrieve_documents(query1, documents, target_word)
print()
retrieve_documents(query2, documents, target_word)

