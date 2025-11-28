from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import json
import os
import re
import numpy as np
import joblib


_space_re = re.compile(r"\s+")


def preprocess_text(s: str) -> str:
    """Lowercase, strip, underscore->space, whitespace collapse."""
    s = (s or "").replace("_", " ")
    s = _space_re.sub(" ", s).strip().lower()
    return s


def collect_article_tfidf(articles, use_title=True, min_len=20):
    """Convert list of articles to normalized, de-duplicated passages with ids."""
    collected = []
    seen_keys = set()
    for idx, art in enumerate(articles):
        raw_title = art.get("title", "").strip()
        raw_text = art.get("text", "").strip()
        full_raw = (raw_title + " " + raw_text) if use_title else raw_text
        full_norm = preprocess_text(full_raw)
        title_norm = preprocess_text(raw_title)
        if len(full_norm) < min_len:
            continue

        text_hash = hashlib.sha1(full_norm.encode("utf-8")).hexdigest()
        dedupe_key = (title_norm, text_hash)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        passage_id = hashlib.sha1(f"{title_norm}|{idx}".encode("utf-8")).hexdigest()
        collected.append(
            {
                "id": passage_id,
                "title": raw_title,
                "title_norm": title_norm,
                "text": raw_text,
                "text_norm": full_norm,
                "text_hash": text_hash,
                "original_index": idx,
            }
        )
    return collected

def train_tfidf(collected_articles, max_features=50000):
      #Fit TF-IDF on normalized texts aligned with passage_order.
    texts = []
    passage_order = []
    for item in collected_articles:
        if isinstance(item, dict):
            texts.append(item.get("text_norm", ""))
            passage_order.append(item.get("id"))
        elif isinstance(item, tuple):
            texts.append(item[1])
            passage_order.append(None)
        else:
            texts.append(str(item))
            passage_order.append(None)
    
    vectorizer = TfidfVectorizer(
        decode_error='replace',
        strip_accents='unicode',
        stop_words='english',
        max_df=0.9,
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        lowercase=False,  # already normalized
    )
    tfidf_matrix = vectorizer.fit_transform(texts) 
    return vectorizer, tfidf_matrix, passage_order


# Add it optional - save and load functions 
def save_artifacts_tfidf(vectorizer, tfidf_matrix, metadata, passage_order, prefix="tfidf_wiki"):

    os.makedirs('question4', exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix}, f'question4/{prefix}_artifacts.pkl')
    with open(f'question4/{prefix}_metadata.json', 'w', encoding='utf-8') as f:
        json.dump({"metadata": metadata, "passage_order": passage_order}, f, ensure_ascii=False)

def load_artifacts_tfidf(prefix="tfidf_wiki"):
    try:
        data = joblib.load(f'question4/{prefix}_artifacts.pkl')
    except FileNotFoundError:
        return None, None, [], []
    
    meta = {"metadata": [], "passage_order": []}
    try:
        with open(f'question4/{prefix}_metadata.json', 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except FileNotFoundError:
        pass
    return data.get('vectorizer'), data.get('tfidf_matrix'), meta.get("metadata", []), meta.get("passage_order", [])


def retrieve_topk_single(query, vectorizer, tfidf_matrix, passage_order, top_k=5):
    """Return sorted list of (passage_id, score)."""
    if vectorizer is None or tfidf_matrix is None or len(passage_order) == 0:
        return []

    q_vec = vectorizer.transform([preprocess_text(query)])
    cosine_sim = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_index = np.argsort(-cosine_sim)[:top_k * 2]  # oversample then stable sort

    scored = []
    for index in top_index:
        if index >= len(passage_order):
            continue
        pid = passage_order[index]
        scored.append((pid, float(cosine_sim[index])))

    scored = sorted(scored, key=lambda x: (-x[1], x[0]))[:top_k]
    return scored
