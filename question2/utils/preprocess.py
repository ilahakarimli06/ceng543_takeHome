#!/usr/bin/env python3
"""
preprocess.py (clean version, no total_lines_est)
- Unicode NFKC normalizasyon
- Smart quote replacement & control character cleaning
- Input: data/raw/{train.en, train.de, val.en, val.de, test.en, test.de} OR train.tsv/csv
- Output: data/cleaned/{train.en, train.de, ...}
"""

import argparse
import logging
import unicodedata
import re
from pathlib import Path
from itertools import islice
from typing import Iterable, Tuple, Iterator, Optional
from tqdm import tqdm
import csv

# --------------------------
# Normalization helpers
# --------------------------
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\u2060\uFEFF]")  # suggested by ChatGPT
THIN_SPACES_RE = re.compile(r"[\u2000-\u200A\u202F\u205F]")  # suggested by ChatGPT

SMART_QUOTES_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00A0": " ", "\u2026": "...",
    "\u00A0": " ", "\u2026": "...",
    "\u201E": '"', "\u201A": "'",                 # Almanca alçak tırnaklar
    "\u00AB": '"', "\u00BB": '"',                 # « »
    "\u2039": "'", "\u203A": "'"                  # ‹ ›
}

def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for k, v in SMART_QUOTES_MAP.items():
        s = s.replace(k, v)

        # görünmezler ve ince boşluklar
    s = ZERO_WIDTH_RE.sub("", s)
    s = THIN_SPACES_RE.sub(" ", s)
    s = CONTROL_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --------------------------
# IO helpers
# --------------------------
def read_parallel_files(input_dir: Path):
    """Yield known split names (train/validation/test) with .en/.de files."""
    for split in ["train", "validation", "test"]:
        en = input_dir / f"{split}.en"
        de = input_dir / f"{split}.de"
        if en.exists() and de.exists():
            yield split, en, de

def open_parallel_iterators(src_path: Path, tgt_path: Path, encoding="utf-8") -> Tuple[Iterator[str], Iterator[str]]:
    def iter_file(p: Path):
        with p.open("r", encoding=encoding) as f:
            for line in f:
                yield line.rstrip("\r\n")
    return iter_file(src_path), iter_file(tgt_path)


def chunked_pair_iterator(src_iter, tgt_iter, chunk_size: int):
    while True:
        src_chunk = list(islice(src_iter, chunk_size))
        tgt_chunk = list(islice(tgt_iter, chunk_size))
        if not src_chunk and not tgt_chunk:
            break
        n = min(len(src_chunk), len(tgt_chunk))
        if n == 0:  # biri bittiğinde sonsuz döngüyi önle
            break
        yield src_chunk[:n], tgt_chunk[:n]
        if n < chunk_size:
            break

def write_lines_append(path: Path, lines: Iterable[str], encoding="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding=encoding) as f:
        for line in lines:
            f.write(line + "\n")

# --------------------------
# Main
# --------------------------
MAX_CHARS = 1000
MIN_CHARS = 2

def keep_pair(en_sentence: str, de_sentence: str) -> bool:
    if len(en_sentence) < MIN_CHARS or len(de_sentence) < MIN_CHARS:
        return False
    if len(en_sentence) > MAX_CHARS or len(de_sentence) > MAX_CHARS:
        return False
    ratio = (len(en_sentence) + 1) / (len(de_sentence) + 1)
    return 0.3 <= ratio <= 3.0


def main(args):
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s: %(message)s")
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(read_parallel_files(input_dir))
    if not pairs:
        raise RuntimeError(f"No parallel files found in {input_dir}")
    logging.info(f"Found splits: {[p[0] for p in pairs]}")

    for split, src_path, tgt_path in pairs:
        logging.info(f"Processing {split}: {src_path}, {tgt_path}")
        src_iter, tgt_iter = open_parallel_iterators(src_path, tgt_path, encoding=args.encoding)

        out_src = out_dir / f"{split}.en"
        out_tgt = out_dir / f"{split}.de"
        for p in (out_src, out_tgt):
            if p.exists():
                p.unlink()

        processed = 0
        for src_chunk, tgt_chunk in tqdm(
            chunked_pair_iterator(src_iter, tgt_iter, args.chunksize), desc=f"{split} chunks"
        ):
            norm_src = []
            norm_tgt = []
            for s, t in zip(src_chunk, tgt_chunk):
                s2 = normalize_text(s)
                t2 = normalize_text(t)
                if keep_pair(s2, t2):
                    norm_src.append(s2)
                    norm_tgt.append(t2)

            if not norm_src:
                continue

            write_lines_append(out_src, norm_src, encoding=args.encoding)
            write_lines_append(out_tgt, norm_tgt, encoding=args.encoding)

            processed += len(norm_src)
            logging.info(f"{split}: processed {processed} lines so far")

        logging.info(f"✅ Wrote cleaned: {out_src}, {out_tgt}")

    logging.info("All done!")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess raw bilingual data (normalization only)")
    p.add_argument("--input_dir", type=str, required=True, help="path to data/raw")
    p.add_argument("--out_dir", type=str, required=True, help="path to data/cleaned")
    p.add_argument("--chunksize", type=int, default=20000)
    p.add_argument("--encoding", type=str, default="utf-8")
    p.add_argument("--log_level", type=str, default="INFO")
    args = p.parse_args()
    main(args)
