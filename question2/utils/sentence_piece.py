#!/usr/bin/env python3

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import sentencepiece as spm


def _collect(pattern: str) -> List[str]:
    files = sorted(Path(".").glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matched pattern: {pattern}")
    return [str(p) for p in files]


def train(
    input_glob: str,
    model_prefix: str,
    vocab_size: int = 8000,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    input_sentence_size: int = 1_000_000,
    shuffle_input_sentence: bool = False,
    user_defined_symbols: Iterable[str] = ("<pad>",),
    pad_id: Optional[int] = None,
) -> None:
    params = [
        f"--input={','.join(_collect(input_glob))}",
        f"--model_prefix={model_prefix}",
        f"--model_type={model_type}",
        f"--vocab_size={vocab_size}",
        f"--character_coverage={character_coverage}",
        f"--input_sentence_size={input_sentence_size}",
        f"--shuffle_input_sentence={str(shuffle_input_sentence).lower()}",
    ]
    symbols = [s for s in user_defined_symbols if s]
    if pad_id is not None and pad_id >= 0:
        symbols = [s for s in symbols if s != "<pad>"]
    if symbols:
        params.append(f"--user_defined_symbols={','.join(symbols)}")
    if pad_id is not None and pad_id >= 0:
        params.append(f"--pad_id={pad_id}")

    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.Train(" ".join(params))
    print(f"saved {model_prefix}.model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--model_prefix", required=True)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--model_type", default="unigram")
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument("--input_sentence_size", type=int, default=1_000_000)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--user_defined_symbols", default="<pad>")
    parser.add_argument("--pad_id", type=int, default=-1)
    parser.add_argument("--help", action="help")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        input_glob=args.input_glob,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=args.shuffle,
        user_defined_symbols=[s.strip() for s in args.user_defined_symbols.split(",") if s.strip()],
        pad_id=args.pad_id if args.pad_id >= 0 else None,
    )


if __name__ == "__main__":
    main()