import os
from datasets import load_dataset


def load_hotpotqa_simple(num_samples=1000, split='train'):

    print(f"Loading HotpotQA dataset ({split}, {num_samples} samples)...")
    
    # HuggingFace token to avoid rate limits
    token = "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"
    dataset = load_dataset('hotpot_qa', 'fullwiki', split=split, token=token)
    
    articles = []
    questions = []
    seen_titles = set()
    
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        
        #  Add articles from context
        for title, sentences in zip(example['context']['title'], example['context']['sentences']):
            if title not in seen_titles:
                text = ' '.join(sentences)
                articles.append({
                    'title': title,
                    'text': text
                })
                seen_titles.add(title)
        
        # Save question and answer
        questions.append({
            'query': example['question'],
            'answer': example['answer'],
            'supporting_facts': example['supporting_facts']
        })
        
        if (i + 1) % 100 == 0:
            print(f"{i + 1} samples loaded...")
    
    print(f"Total: {len(articles)} articles, {len(questions)} questions loaded!")
    return articles, questions
