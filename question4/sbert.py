from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import os

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# Same with tf_idf.py's collect_article
def collect_article_sbert(articles, use_title=True, min_len=20):
    """Convert list of articles to a flat list of (title, text) tuples."""
    collected = []
    for art in articles:
        title = art.get('title', '').strip()
        text = art.get('text', '').strip()
        full_text = (title + ' ' + text) if use_title else text
        if len(full_text) < min_len:
            continue
        # Store as tuple: (title, full_text)
        collected.append((title, full_text))
    return collected


def train_sbert(collected_articles, model_name=DEFAULT_MODEL_NAME, batch_size=32):
    # Extract texts from (title, text) tuples
    texts = [item[1] if isinstance(item, tuple) else item for item in collected_articles]
    
    model = SentenceTransformer(model_name)
    # convert_to_numpy=True -> return numpy array (float32)
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    # embeddings shape: (n_docs, dim)
    return model, embeddings


def save_artifacts_sbert(model, embeddings, collected, prefix="sbert_wiki"):
    
    # Get model name safely
    try:
        if hasattr(model, "_first_module"):
            first_module = model._first_module()
            model_name = getattr(first_module, "config", None)
            if model_name and hasattr(model_name, "name_or_path"):
                model_name = model_name.name_or_path
            else:
                model_name = None
        else:
            model_name = None
    except:
        model_name = None

    os.makedirs('question4', exist_ok=True)
    with open(f'question4/{prefix}_artifacts.pkl', 'wb') as f:
        pickle.dump({
            "model_name": model_name,
            "embeddings": embeddings
        }, f)
    # Save with titles (format: TITLE|||TEXT)
    with open(f'question4/{prefix}_collected.txt', 'w', encoding='utf-8') as f:
        for item in collected:
            if isinstance(item, tuple):
                title, text = item
                f.write(f"{title}|||{text.replace('\n', ' ')}\n")
            else:
                f.write(item.replace('\n', ' ') + '\n')

def load_artifacts_sbert(prefix="sbert_wiki"):
    """Load embeddings and collected texts. Returns (model_name or None, embeddings or None, collected list)."""
    model_name = None
    embeddings = None
    try:
        with open(f'question4/{prefix}_artifacts.pkl', 'rb') as f:
            data = pickle.load(f)
            model_name = data.get("model_name")
            embeddings = data.get("embeddings")
    except FileNotFoundError:
        pass

    # Load with titles (format: TITLE|||TEXT)
    collected = []
    try:
        with open(f'question4/{prefix}_collected.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if '|||' in line:
                    parts = line.split('|||', 1)
                    collected.append((parts[0], parts[1]))
                else:
                    # Backward compatibility
                    collected.append(('', line))
    except FileNotFoundError:
        pass

    return model_name, embeddings, collected


def retrieve_sbert(query, model, embeddings, collected, top_k=5):
    """
    Same logic with tf-idf retrieve, but using SBERT embeddings.
    """
    if embeddings is None or len(embeddings) == 0:
        return []

    # encode query
    q_emb = model.encode([query], convert_to_numpy=True)
    # cosine similarity
    cosine_sim= cosine_similarity(q_emb, embeddings).flatten()  
    top_index = np.argsort(-cosine_sim)[:top_k]
    
    results = []
    for index in top_index:
        item = collected[index]
        if isinstance(item, tuple):
            title, text = item
            # Return: (index, score, title, text)
            results.append((int(index), float(cosine_sim[index]), title, text))
        else:
            # Backward compatibility
            results.append((int(index), float(cosine_sim[index]), '', item))
    return results