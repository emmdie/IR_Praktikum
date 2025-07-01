import torch
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer, BertForMaskedLM
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Mehrere Tokens erlaubt :))) 
def validate_target_word(target_word):
    word_pieces = tokenizer.tokenize(target_word)
    if len(word_pieces) != 1:
        raise ValueError(f"Target word '{target_word}' splits into {word_pieces}, which is not 1 token long.")
    return word_pieces[0]

def get_embedding(sentence, target_token):
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    try:
        token_index = tokens.index(target_token)
    except ValueError:
        raise ValueError(f"Token '{target_token}' not found in sentence tokens: {tokens}")

    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_size)

    return token_embeddings[token_index]  

def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def visualize_embeddings(sentences, target_word):
    target_token = validate_target_word(target_word)

    contextual_embeddings = []
    for sentence in sentences:
        emb = get_embedding(sentence, target_token)
        contextual_embeddings.append(emb)

    standalone_emb = get_embedding(target_word, target_token)

    for i, emb in enumerate(contextual_embeddings):
        sim = cosine_similarity(standalone_emb, emb)
        print(f"Cosine similarity (Standalone ↔ Sentence {i+1}): {sim:.4f}")

    all_embeddings = contextual_embeddings + [standalone_emb]
    emb_matrix = torch.stack(all_embeddings).numpy()

    plt.figure(figsize=(18, 3))
    sns.heatmap(emb_matrix, cmap="viridis", cbar=True)
    plt.yticks([0.5, 1.5, 2.5], ["Sentence 1", "Sentence 2", "Standalone"])
    plt.xlabel("Embedding Dimension")
    plt.title(f"Heatmap of BERT Embeddings for '{target_word}' (Single-Token)")
    plt.tight_layout()
    plt.show()

sentences = ["The road winds through the scenic, mountainous country of New Hampshire north of the White Mountain National Forest.", "Some areas formed in sand dunes swept by winds from the Connecticut River Valley as ancient glacial Lake Hitchcock receded."]
target_word = "winds"

visualize_embeddings(sentences, target_word)

