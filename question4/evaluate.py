from flan_t5_rag import rag_with_t5_tfidf, rag_with_t5_sbert, _generate_with_t5
from load_hotpotqa import load_hotpotqa_simple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tf_idf import preprocess_text
import numpy as np
import pandas as pd
import json
import os

def evaluate_retrieval(results, supporting_titles, k=3):
    if not results or not supporting_titles:
        return 0.0, 0.0

    # Normalize gold titles
    gold = set()
    for s in supporting_titles:
        if isinstance(s, (list, tuple)) and s:
            gold.add(preprocess_text(s[0]))
        else:
            gold.add(preprocess_text(str(s)))
    gold = {g for g in gold if g}
    
    if not gold:
        return 0.0, 0.0

    # Get top-k retrieved titles
    top_k = results[:min(k, len(results))]
    retrieved = set()
    for r in top_k:
        title = r[2] if len(r) >= 3 else ""
        title_norm = preprocess_text(str(title))
        if title_norm:
            retrieved.add(title_norm)
    
    # Check for partial matches (important for multi-word titles)
    found = 0
    matched_gold = set()
    for rt in retrieved:
        for gt in gold:
            # Exact match or one contains the other
            if rt == gt or rt in gt or gt in rt:
                found += 1
                matched_gold.add(gt)
                break
    
    precision = found / len(retrieved) if retrieved else 0.0
    recall = len(matched_gold) / len(gold) if gold else 0.0
    return precision, recall


def evaluate_generation(predicted, reference):
    if not predicted or not reference:
        return 0.0, 0.0, 0.0
    
    pred_clean = str(predicted).strip()
    ref_clean = str(reference).strip()
    
    if not pred_clean or not ref_clean:
        return 0.0, 0.0, 0.0

    # BLEU-1 (unigram only for short answers)
    ref_tokens = word_tokenize(ref_clean.lower())
    pred_tokens = word_tokenize(pred_clean.lower())
    smoothing = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)

    # ROUGE-L - case-insensitive via lowercase
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(ref_clean.lower(), pred_clean.lower())['rougeL'].fmeasure

    # BERTScore - use original case for semantic similarity (case-sensitive)
    try:
        _, _, F1 = bert_score([pred_clean], [ref_clean], lang='en', 
                             rescale_with_baseline=False, verbose=False)
        bertscore_f1 = float(F1[0])
    except Exception:
        bertscore_f1 = 0.0

    return bleu, rouge_l, bertscore_f1


def select_faithful_vs_hallucinated(results, top_n=3):
    df = pd.DataFrame(results)

    # Verify columns
    required_cols = {'query', 'reference', 'answer', 'bleu', 'bertscore'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"Warning: Missing columns in results: {missing}")
        return None, None

    # Optional: include ROUGE if available
    if 'rouge' in df.columns:
        df['joint'] = (df['bleu'] + df['rouge'] + df['bertscore']) / 3
    else:
        df['joint'] = (df['bleu'] + df['bertscore']) / 2

    faithful = df.sort_values('joint', ascending=False).head(top_n)
    hallucinated = df.sort_values('joint').head(top_n)

    print("\n" + "="*70)
    print("FAITHFUL EXAMPLES (High Quality)")
    print("="*70)
    for _, row in faithful.iterrows():
        print(f"\nQ: {row['query']}")
        print(f"Reference: {row['reference']}")
        print(f"Generated: {row['answer']}")
        print(f"Metrics: BLEU={row['bleu']:.3f}, BERT={row['bertscore']:.3f}"
              + (f", ROUGE={row['rouge']:.3f}" if 'rouge' in row else ""))

    print("\n" + "="*70)
    print("HALLUCINATED EXAMPLES (Low Quality)")
    print("="*70)
    for _, row in hallucinated.iterrows():
        print(f"\nQ: {row['query']}")
        print(f"Reference: {row['reference']}")
        print(f"Generated: {row['answer']}")
        print(f"Metrics: BLEU={row['bleu']:.3f}, BERT={row['bertscore']:.3f}"
              + (f", ROUGE={row['rouge']:.3f}" if 'rouge' in row else ""))
    print("="*70 + "\n")

    return faithful, hallucinated


def run_evaluation(num_test=50):
    """Tam evaluation - 3 mod: Retrieval-only, Generation-only (gold), Joint RAG"""
    # Load BOTH articles and questions from the same split
    articles, questions = load_hotpotqa_simple(num_samples=num_test, split='validation')
    
    # Build title - article mapping for faster lookup
    article_map = {}
    for art in articles:
        title_norm = preprocess_text(art.get('title', ''))
        if title_norm:
            article_map[title_norm] = art

    # Create output directory
    os.makedirs('question4/output', exist_ok=True)

    methods = {
        "TF-IDF + T5": rag_with_t5_tfidf,
        "SBERT + T5": rag_with_t5_sbert
    }
    
    all_method_results = {}

    for method_name, rag_func in methods.items():
        print(f"\n=== METHOD: {method_name} ===\n")
        retrieval_precisions, retrieval_recalls = [], []
        gen_only_bleu, gen_only_rouge, gen_only_bert = [], [], []
        joint_bleu, joint_rouge, joint_bert = [], [], []
        all_results = []  # For qualitative analysis

        for i, test in enumerate(questions):
            print(f"Question {i+1}/{len(questions)}: {test['query']}")
            supporting_titles = [sf[0] for sf in test['supporting_facts']]

            # call rag once - use top_k=5 for better coverage
            answer, results = rag_func(test['query'], top_k=5)

            # RETRIEVAL-ONLY - evaluate on top-5 (consistent with retrieval)
            precision, recall = evaluate_retrieval(results[:5], supporting_titles, k=5)
            retrieval_precisions.append(precision); retrieval_recalls.append(recall)

            # GENERATION-ONLY (gold context from supporting facts)
            gold_context = ""
            for title in supporting_titles:
                title_norm = preprocess_text(str(title))
                # Try exact match first
                found = False
                if title_norm in article_map:
                    a = article_map[title_norm]
                    text = a.get('text', '')
                    if text:
                        gold_context += f"{a['title']}: {text[:500]}\n\n"
                        found = True
                
                # Try partial match if exact match fails
                if not found:
                    for art_title_norm, a in article_map.items():
                        if title_norm in art_title_norm or art_title_norm in title_norm:
                            text = a.get('text', '')
                            if text:
                                gold_context += f"{a['title']}: {text[:500]}\n\n"
                                break
            
            if gold_context.strip():
                # Limit context length to avoid exceeding model input limits (~512 tokens for FLAN-T5)
                gold_context = gold_context[:1500]  # Safe character limit (~400 tokens)
                gold_answer = _generate_with_t5(gold_context, test['query'])
                bleu, rouge_l, bertscore_f1 = evaluate_generation(gold_answer, test['answer'])
                gen_only_bleu.append(bleu)
                gen_only_rouge.append(rouge_l)
                gen_only_bert.append(bertscore_f1)
            else:
                gen_only_bleu.append(0.0)
                gen_only_rouge.append(0.0)
                gen_only_bert.append(0.0)

            # JOINT RAG (retrieved context -> generation already produced as `answer`)
            bleu_j, rouge_j, bert_j = evaluate_generation(answer, test['answer'])
            joint_bleu.append(bleu_j); joint_rouge.append(rouge_j); joint_bert.append(bert_j)

            # Collect for qualitative analysis
            # Format: (idx, score, title, text)
            retrieved_context = ""
            for r in results:
                title = r[2] if len(r) > 2 else "Unknown"
                text = r[3] if len(r) > 3 else ""
                retrieved_context += f"{title}: {text[:200]}... "
            
            all_results.append({
                'query': test['query'],
                'reference': test['answer'],
                'answer': answer,
                'retrieved_context': retrieved_context.strip(),
                'topk': [{'pid': r[0], 'score': r[1], 'title': r[2] if len(r) > 2 else ''} for r in results],
                'bleu': bleu_j,
                'rouge': rouge_j,
                'bertscore': bert_j
            })

            print(f"  Reference: {test['answer']}")
            print(f"  Retrieval-only P@5: {precision:.3f}, R@5: {recall:.3f}")
            if gen_only_bleu:
                print(f"  Generation-only (gold) BLEU: {gen_only_bleu[-1]:.3f}, ROUGE-L: {gen_only_rouge[-1]:.3f}, BERT: {gen_only_bert[-1]:.3f}")
            print(f"  Joint RAG BLEU: {bleu_j:.3f}, ROUGE-L: {rouge_j:.3f}, BERT: {bert_j:.3f}\n")

        print(f"\n--- AVERAGES ({method_name}) ---")
        print(f"Retrieval: Precision@5={np.mean(retrieval_precisions) if retrieval_precisions else 0.0:.3f}, Recall@5={np.mean(retrieval_recalls) if retrieval_recalls else 0.0:.3f}")
        if gen_only_bleu:
            print(f"Generation-only (gold): BLEU={np.mean(gen_only_bleu):.3f}, ROUGE-L={np.mean(gen_only_rouge):.3f}, BERT={np.mean(gen_only_bert):.3f}")
        print(f"Joint RAG: BLEU={np.mean(joint_bleu) if joint_bleu else 0.0:.3f}, ROUGE-L={np.mean(joint_rouge) if joint_rouge else 0.0:.3f}, BERT={np.mean(joint_bert) if joint_bert else 0.0:.3f}")
        print("="*40 + "\n")

        # d) Qualitative analysis: faithful vs hallucinated
        faithful_df, hallucinated_df = select_faithful_vs_hallucinated(all_results, top_n=5)

        # Serialize all columns (küçültmeden) ama None durumda boş liste
        faithful_list = faithful_df.to_dict(orient='records') if faithful_df is not None else []
        hallucinated_list = hallucinated_df.to_dict(orient='records') if hallucinated_df is not None else []

        # Save results to JSON
        all_method_results[method_name] = {
            "retrieval": {
                "precision_at_5": float(np.mean(retrieval_precisions)) if retrieval_precisions else 0.0,
                "recall_at_5": float(np.mean(retrieval_recalls)) if retrieval_recalls else 0.0
            },
            "generation_only": {
                "bleu": float(np.mean(gen_only_bleu)) if gen_only_bleu else 0.0,
                "rouge_l": float(np.mean(gen_only_rouge)) if gen_only_rouge else 0.0,
                "bertscore": float(np.mean(gen_only_bert)) if gen_only_bert else 0.0
            },
            "joint_rag": {
                "bleu": float(np.mean(joint_bleu)) if joint_bleu else 0.0,
                "rouge_l": float(np.mean(joint_rouge)) if joint_rouge else 0.0,
                "bertscore": float(np.mean(joint_bert)) if joint_bert else 0.0
            },
            "examples": all_results[:10],  # Save top 10 examples
            "faithful_examples": faithful_list,
            "hallucinated_examples": hallucinated_list
        }
    
    # Save all results to JSON
    with open('question4/output/rag_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_method_results, f, indent=2, ensure_ascii=False)
    print("\nResults saved to question4/output/rag_evaluation_results.json")


if __name__ == "__main__":
    run_evaluation(num_test=50)
