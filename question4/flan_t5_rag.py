from load_hotpotqa import load_hotpotqa_simple
from tf_idf import (
    collect_article_tfidf,
    train_tfidf,
    save_artifacts_tfidf,
    load_artifacts_tfidf,
    retrieve_topk_single,
)
from sbert import (
    collect_article_sbert,
    train_sbert,
    save_artifacts_sbert,
    load_artifacts_sbert,
    retrieve_sbert,
)
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Extract entities for better retrieval
import re


# Load FLAN-T5 once and cache for reuse
_t5_cache = {"tokenizer": None, "model": None, "device": None}


def _get_flan_t5():
    if _t5_cache["model"] is not None:
        return _t5_cache["tokenizer"], _t5_cache["model"], _t5_cache["device"]

    model_name = "google/flan-t5-base"  # Better QA quality than t5-small
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    _t5_cache.update({"tokenizer": tokenizer, "model": model, "device": device})
    return tokenizer, model, device


def _generate_with_t5(context, query):
    """FLAN-T5 short-form answer generation with single model load."""
    tokenizer, model, device = _get_flan_t5()

    prompt = (
        "Answer this question briefly in few words based on the context.\n\n"
        f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    out = model.generate(
        **inputs,
        max_length=15,  # Short answers (1-3 words typical for HotpotQA)
        num_beams=5,
        do_sample=False,
        early_stopping=True,
        repetition_penalty=1.2,  # Avoid repetition
        length_penalty=1.3,  # Prefer shorter answers
    )

    answer = tokenizer.decode(out[0], skip_special_tokens=True)
    return answer


def rag_with_t5_tfidf(query, top_k=3, max_features=50000):
    # Download HotpotQA articles from train split (avoid data leakage)
    articles, _ = load_hotpotqa_simple(num_samples=5000, split="train")
    collected = collect_article_tfidf(articles, use_title=True, min_len=10)

    # TF-IDF load or train
    vectorizer, tfidf_matrix, saved_metadata, passage_order = load_artifacts_tfidf(
        prefix="tfidf_wiki"
    )
    if vectorizer is None or tfidf_matrix is None or len(saved_metadata) == 0:
        vectorizer, tfidf_matrix, passage_order = train_tfidf(collected, max_features=max_features)
        save_artifacts_tfidf(vectorizer, tfidf_matrix, collected, passage_order, prefix="tfidf_wiki")
        saved_metadata = collected

    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
    # Boost query with entities
    enhanced_query = query + " " + " ".join(entities)

    # Retriever with enhanced query
    topk = retrieve_topk_single(
        enhanced_query, vectorizer, tfidf_matrix, passage_order, top_k=top_k
    )
    # results -> list of (pid, score)

    # Map pid -> metadata
    pid2meta = {m["id"]: m for m in saved_metadata if isinstance(m, dict) and "id" in m}
    results = []
    for pid, score in topk:
        meta = pid2meta.get(pid, {})
        results.append((pid, score, meta.get("title", ""), meta.get("text", "")))

    # Join context with titles
    context = ""
    for r in results:
        title = r[2] if len(r) > 2 else "Unknown"
        text = r[3] if len(r) > 3 else ""
        # Limit text length to avoid prompt overflow
        text_preview = text[:300] if len(text) > 300 else text
        context += f"{title}: {text_preview}\n\n"

    # Generate answer with T5
    answer = _generate_with_t5(context, query)

    return answer, results


def rag_with_t5_sbert(query, top_k=3, max_features=50000):
    # Download HotpotQA articles from train split (avoid data leakage)
    articles, _ = load_hotpotqa_simple(num_samples=5000, split="train")
    collected = collect_article_sbert(articles, use_title=True, min_len=10)

    # SBERT load or train
    sbert_model_name, embeddings, saved_collected = load_artifacts_sbert(
        prefix="sbert_wiki"
    )
    if sbert_model_name is None or embeddings is None or len(saved_collected) == 0:
        sbert_model, embeddings = train_sbert(collected)
        save_artifacts_sbert(sbert_model, embeddings, collected, prefix="sbert_wiki")
        saved_collected = collected
    else:
        sbert_model = SentenceTransformer(
            sbert_model_name if sbert_model_name else "all-MiniLM-L6-v2"
        )

    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
    enhanced_query = query + " " + " ".join(entities)

    # Retriever with enhanced query
    results = retrieve_sbert(
        enhanced_query, sbert_model, embeddings, saved_collected, top_k=top_k
    )
    # results -> list of (idx, score, title, text)

    # Join context with titles
    context = ""
    for r in results:
        title = r[2] if len(r) > 2 else "Unknown"
        text = r[3] if len(r) > 3 else ""
        # Limit text length to avoid prompt overflow
        text_preview = text[:300] if len(text) > 300 else text
        context += f"{title}: {text_preview}\n\n"

    # Generate answer with T5
    answer = _generate_with_t5(context, query)

    return answer, results
