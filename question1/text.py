from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

ds = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

lengths = []
for i, text in enumerate(ds["train"]["text"]):
    tokens = tokenizer(text, truncation=False)["input_ids"]
    lengths.append(len(tokens))
    if i % 1000 == 0:
        print(f"Processed {i} samples...")

print(f"Mean: {np.mean(lengths):.1f}")
print(f"Median: {np.median(lengths):.1f}")
print(f"95th percentile: {np.percentile(lengths, 95):.1f}")
print(f"% truncated at 256: {(np.array(lengths) > 256).mean() * 100:.1f}%")
