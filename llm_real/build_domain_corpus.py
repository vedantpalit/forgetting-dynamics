"""A domain corpus for continued pretraining, in the trainer's {docs, held} format.

Domain-adaptive continued pretraining (medical, code, legal) is the most common real update a
practitioner runs, and the setting where the continual-pretraining literature reports a transient
drop and recovery of general performance (the "stability gap"). This script only fetches and
samples text; nothing is designed. Run it with network access, then train offline:

  .venv-llm/bin/python -m llm_real.build_domain_corpus --dataset <hf id> [--config C] [--split S]
      --text_field text --n_docs 20000 --out llm_real/out/corpus_pubmed.json

Documents shorter than --min_chars are dropped; a seeded 90/10 split gives the held-out set
(the same split rule as corpus.py). Token counts are reported so runs can be sized: the trainer
uses tokens_per_step = 32,768, so N steps consume N x 32,768 tokens; with fewer tokens than that
in the corpus the run cycles through it (epochs are reported at train time).
"""
import argparse
import json
import os
import random

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_files", default=None,
                    help="raw file(s) inside the repo or an hf:// URL; bypasses dataset scripts, "
                         "which datasets >= 4 refuses")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--n_docs", type=int, default=20000)
    ap.add_argument("--min_chars", type=int, default=400)
    ap.add_argument("--max_chars", type=int, default=6000, help="truncate long docs (code files)")
    ap.add_argument("--chunk", action="store_true",
                    help="split long docs into max_chars pieces at paragraph breaks instead of truncating "
                         "(for corpora of a few huge documents, e.g. CFR titles)")
    ap.add_argument("--filter_field", default=None, help="keep rows where this field == --filter_value")
    ap.add_argument("--filter_value", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    kw = {"data_files": a.data_files} if a.data_files else {}
    ds = load_dataset(a.dataset, a.config, split=a.split, streaming=True, **kw)
    docs, seen = [], 0
    for row in ds:
        seen += 1
        if a.filter_field and str(row.get(a.filter_field)) != a.filter_value:
            continue
        t = row.get(a.text_field)
        if not isinstance(t, str):
            continue
        t = t.strip()
        if len(t) < a.min_chars:
            continue
        pieces = [t[:a.max_chars]]
        if a.chunk and len(t) > a.max_chars:
            pieces, buf = [], ""
            for para in t.split("\n\n"):
                if len(buf) + len(para) + 2 > a.max_chars and buf:
                    pieces.append(buf); buf = ""
                buf = (buf + "\n\n" + para) if buf else para
                if len(buf) > a.max_chars:                 # a single over-long paragraph
                    pieces.append(buf[:a.max_chars]); buf = ""
            if buf:
                pieces.append(buf)
        for piece in pieces:
            if len(piece) >= a.min_chars:
                docs.append({"text": piece})
        if len(docs) >= a.n_docs:
            break
        if seen % 20000 == 0:
            print(f"  scanned {seen}, kept {len(docs)}", flush=True)
    rng.shuffle(docs)
    k = max(1, len(docs) // 10)
    held, train = docs[:k], docs[k:]
    chars = sum(len(d["text"]) for d in train)
    meta = {"dataset": a.dataset, "config": a.config, "data_files": a.data_files, "split": a.split,
            "text_field": a.text_field,
            "n_train": len(train), "n_held": len(held), "train_chars": chars,
            "approx_tokens": chars // 4, "seed": a.seed}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"docs": train, "held": held, "meta": meta}, open(a.out, "w", encoding="utf-8"))
    print(f"wrote {a.out}: {len(train)} train docs, {len(held)} held, ~{chars // 4 / 1e6:.1f}M tokens "
          f"(~{chars // 4 / 32768:.0f} steps per epoch at 32,768 tokens/step)")
    print(f"example: {train[0]['text'][:300]!r}")


if __name__ == "__main__":
    main()
