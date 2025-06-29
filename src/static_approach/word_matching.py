from rapidfuzz import process, fuzz
from sentence_transformers import util

def fuzzy_match(word, candidates, threshold=85):
    match, score, _ = process.extractOne(
        query=word,
        choices=candidates,
        scorer=fuzz.ratio
    )
    return match if score >= threshold else None

def embedding_match(word, candidates, model, sim_threshold=0.6, cache=None):
    query_vec = model.encode([word])[0]
    
    best_score = -1
    best_word = None

    for candidate in candidates:
        if cache is not None:
            if candidate not in cache:
                cache[candidate] = model.encode(word, convert_to_tensor=True, normalize_embeddings=True)
            vec = cache[candidate]
        else:
            vec = model.encode(candidate, convert_to_tensor=True, normalize_embeddings=True)

        score = util.cos_sim(query_vec, vec)[0]

        if score > best_score:
            best_score = score
            best_word = candidate

    return best_word if best_score >= sim_threshold else None
