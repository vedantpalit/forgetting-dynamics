"""Local checks for llm_real that need no model: corpus parsing on the fetched people,
template stems, the token packer and mixer (fake tokenizer), the format audit, and the
config flags. Run: python3 -m tests.test_llm_real"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_real import corpus, templates, tokens
from llm_real.config import RunConfig, add_args, from_args
import argparse


class FakeTok:
    """Whitespace tokenizer with a stable vocabulary, so packing can be tested without torch."""
    def __init__(self): self.v = {}
    def __call__(self, s, add_special_tokens=False):
        ids = [self.v.setdefault(w, len(self.v) + 1) for w in s.replace("\n", " \n ").split(" ") if w]
        return type("E", (), {"input_ids": ids})()
    def decode(self, ids):
        inv = {i: w for w, i in self.v.items()}
        return " ".join(inv.get(int(i), "?") for i in ids)


def test_templates():
    for attr, ts in templates.TEMPLATES.items():
        assert len(ts) == 15, attr
        for t in ts:
            assert t.rstrip(".").endswith("{v}"), t
            s = templates.render(t, "Ada Lovelace", "actor")
            assert "{" not in s and "?" not in s
    assert templates.render("{name} was {a} {v}.", "X", "actor") == "X was an actor."
    print("  ok  templates: 15 per attribute, value last, article rendering")


def test_corpus():
    p = "llm_real/out/people_raw.json"
    if not os.path.exists(p):
        print("  skip corpus (no people_raw.json)"); return
    rows = json.load(open(p))
    facts = [corpus.person_facts(r) for r in rows]
    n = sum(1 for f in facts if f)
    assert n > 1000, n
    for f in facts:
        for attr, v in f.items():
            assert attr in templates.ATTRS and 3 <= len(v) <= 40 and not any(c.isdigit() for c in v), (attr, v)
    assert corpus.normalize_value("birth_place", "July 27, 1896 Meriwether County") == "Meriwether County"
    assert corpus.normalize_value("birth_place", "Samuel Foard 12 May 1820 Bohemia Manor, Maryland") == "Bohemia Manor"
    assert corpus.normalize_value("alma_mater", "B.A") is None
    assert corpus.normalize_value("occupation", "Islamic scholar, Muffassir") == "islamic scholar"
    print(f"  ok  corpus: {n} of {len(rows)} people yield facts; normalization cases")


def test_tokens_and_audit():
    tok = FakeTok()
    b = {"people": [{"name": "Ada Lovelace", "prose": "Ada Lovelace was a mathematician who lived in London.",
                     "facts": {"birth_place": "London", "occupation": "mathematician"}},
                    {"name": "Bob Ray", "prose": "Bob Ray played football for years.",
                     "facts": {"employer": "Leeds United"}}],
         "train_templates": {a: templates.split(a)[0] for a in templates.ATTRS}}
    r = tokens.BRenderer(b, tok, augment=True, seed=0)
    g = tokens.TokenPool([tok("generic prose about something else entirely").input_ids] * 5, seed=1)
    sep = tok(tokens.SEP).input_ids
    m = tokens.Mixer(r, g, 0.5, seq_len=40, sep_ids=sep, seed=0)
    batch = m.batch(8)
    assert batch.shape == (8, 40) and batch.dtype == np.int64
    docs = [r.document_string(b, i) for i in range(2)]
    tokens.audit_format(docs, "test B docs", n_show=0)
    try:
        tokens.audit_format(["Q: who? A: me"], "bad", n_show=0)
    except SystemExit:
        pass
    else:
        raise AssertionError("audit accepted QA markup")
    m1 = tokens.Mixer(r, None, 1.0, 40, sep, 0); m1.batch(2)          # ratio 1 needs no generic pool
    try:
        tokens.Mixer(r, None, 0.5, 40, sep, 0)
    except SystemExit:
        pass
    else:
        raise AssertionError("mixer allowed ratio<1 without a generic pool")
    print("  ok  tokens: packing, mixing, augmentation, audit accepts prose and rejects QA")


def test_config():
    ap = add_args(argparse.ArgumentParser())
    ns = ap.parse_args(["--lr", "3e-6", "--b_ratio", "0.1", "--no-augment", "--steps", "100"])
    cfg = from_args(ns)
    assert cfg.lr == 3e-6 and cfg.b_ratio == 0.1 and not cfg.augment and cfg.steps == 100
    assert cfg.name == "real_strict_r0.1_lr3e-06_seed0"
    assert RunConfig().name == "real_aug_r1_lr1e-05_seed0"
    print("  ok  config: flags round-trip, run names")


if __name__ == "__main__":
    print("test_llm_real")
    test_templates(); test_corpus(); test_tokens_and_audit(); test_config()
    print("all passed")
