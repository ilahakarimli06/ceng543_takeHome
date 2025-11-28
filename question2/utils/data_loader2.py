# question2/data_loader2.py
from datasets import load_dataset
from pathlib import Path


def extract_pairs(dataset):
    """Return list of (de, en) pairs from a dataset split."""
    pairs = []
    for example in dataset:

        #HuggingFace-style translation dict
        if "translation" in example:
            trans = example["translation"]
            de = trans.get("de", "").strip()
            en = trans.get("en", "").strip()
        else:
            de = example.get("de", "").strip()
            en = example.get("en", "").strip()
        # Add only valid non-empty pairs
        if de and en:
            pairs.append((de, en))
    return pairs


def write_split(split_name: str, pairs, out_dir: Path):

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # File paths for English and German output
    en_path = out_dir / f"{split_name}.en"
    de_path = out_dir / f"{split_name}.de"

     # Open both files and write aligned lines
    with en_path.open("w", encoding="utf-8") as f_en, de_path.open("w", encoding="utf-8") as f_de:
        for de, en in pairs:
            f_en.write(en + "\n")
            f_de.write(de + "\n")
    print(f"Saved {len(pairs)} pairs to {en_path} and {de_path}")


def main(out_dir="question2/data/raw", repo_id="bentrevett/multi30k"):
    out_dir = Path(out_dir)
    for split in ("train", "validation", "test"):
        print(f"\nDownloading split: {split}")
        ds = load_dataset(repo_id, split=split)
        print("Features:", ds.features)
        print("Length:", len(ds))
        print("Example 0:", ds[0])
        print("Columns:", ds.column_names)

        pairs = extract_pairs(ds)
        write_split(split, pairs, out_dir)


if __name__ == "__main__":
    main()

