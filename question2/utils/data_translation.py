import argparse
import sentencepiece as spm
from pathlib import Path


def encode_file(sp, in_path, out_path, add_bos=True, add_eos=True):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        bos_id = sp.bos_id()
        eos_id = sp.eos_id()
        for line in fin:
            line = line.rstrip("\n")
            ids = sp.encode(line, out_type=int)
            if add_bos and bos_id != -1:
                ids = [bos_id] + ids
            if add_eos and eos_id != -1:
                ids = ids + [eos_id]
            fout.write(" ".join(map(str, ids)) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spm", required=True, help="spm_shared_unigram.model")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--no_bos", action="store_true")
    ap.add_argument("--no_eos", action="store_true")
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm)

    encode_file(sp, args.inp, args.out, add_bos=not args.no_bos, add_eos=not args.no_eos)