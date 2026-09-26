"""Token pools, packing and mixing -- everything between the corpus and the model's input.

WHY PRE-TOKENIZE. llm/inject.py tokenizes every rendered document at every micro-batch (~200
tokenizer calls per optimizer step). Here every fixed string -- each article's prose, each
generic document, and every (template, name, value) sentence -- is tokenized once into an
integer array, and a step is index arithmetic plus concatenation.

WHAT VARIES PER STEP AND WHAT DOES NOT. A B document is the person's real article prose (fixed,
tokenized once) followed, in the augmented arm, by that person's fact sentences in a fresh
random order with a fresh random training phrasing each -- so the facts repeat while the
string does not (the llm/ recipe, which is what made facts learnable rather than strings
memorisable). Sentence order and phrasing are the only randomness; the augmented sentences
are pre-tokenized per (person, attribute, template), so rendering is a table lookup.

FORMAT DISCIPLINE, ENFORCED NOT ASSUMED. A B document must look like the generic documents it
is mixed with: prose paragraphs, no question marks, no list markers, no key-value lines, no
chat or instruction markup. `audit_format` checks every rendered document against those
rules and refuses to proceed on a violation; the run prints a sample of both kinds side by
side so the register can be read, not just asserted.

MIXING. `b_ratio` is the fraction of training ROWS drawn from B; the rest are windows of
generic pre-cutoff text. Rows are packed to `seq_len` from whole documents separated by a
double newline; a document that does not fit is truncated at the row boundary. b_ratio = 1
reproduces llm/'s regime exactly; 0.1 is the practitioner's regime.
"""
import random
import re

import numpy as np

from llm_real.templates import render

SEP = "\n\n"

_BAD = [
    (re.compile(r"\?"), "question mark"),
    (re.compile(r"^\s*[-*•]\s", re.M), "list marker at line start"),
    (re.compile(r"^\s*[A-Za-z ]{1,30}:\s", re.M), "key: value line"),
    (re.compile(r"<\|.*?\|>|\[INST\]|<s>|###|Human:|Assistant:"), "chat or instruction markup"),
    (re.compile(r"^\s*(Q|A|Question|Answer):\s", re.M), "QA marker"),
    (re.compile(r"\n\s*\n\s*\n"), "more than one blank line"),
]


def violation(doc):
    """The first rule a document breaks, or None."""
    for rx, why in _BAD:
        if rx.search(doc):
            return why
    return None


def audit_format(docs, label, n_show=2):
    """Raise on any document that breaks the prose register; print a couple for the eye."""
    bad = []
    for d in docs:
        why = violation(d)
        if why:
            bad.append((why, d[:160]))
    print(f"[format audit] {label}: {len(docs)} documents, {len(bad)} violations")
    for d in docs[:n_show]:
        print("   | " + d[:300].replace("\n", "\n   | "))
    if bad:
        for why, d in bad[:5]:
            print(f"   VIOLATION ({why}): {d!r}")
        raise SystemExit(f"format audit failed for {label}: {len(bad)} documents break the "
                         f"prose register. Fix the corpus, do not train.")


class TokenPool:
    """A list of pre-tokenized documents plus a sampler of packed rows."""

    def __init__(self, docs_ids, seed):
        self.docs = [np.asarray(d, dtype=np.int32) for d in docs_ids if len(d) > 0]
        self.rng = random.Random(seed)
        self.n_tokens = int(sum(len(d) for d in self.docs))

    def row(self, seq_len, sep_ids):
        out = []
        while len(out) < seq_len:
            out.extend(self.docs[self.rng.randrange(len(self.docs))].tolist())
            out.extend(sep_ids)
        return out[:seq_len]


class BRenderer:
    """B documents: article prose (fixed) + augmented fact sentences (fresh order/phrasing)."""

    def __init__(self, b, tok, augment, seed):
        """`b` is the gated B json: people[{name, prose, facts{attr: value}}], train_templates
        {attr: [template]}. Tokenizes prose once and every (person, attr, template) once."""
        self.rng = random.Random(seed)
        self.augment = augment
        self.people = []
        self.templates = b["train_templates"]
        for p in b["people"]:
            prose = tok(p["prose"], add_special_tokens=False).input_ids
            sents = {}
            if augment:
                for attr, v in p["facts"].items():
                    sents[attr] = [tok(" " + render(t, p["name"], v),
                                       add_special_tokens=False).input_ids
                                   for t in self.templates[attr]]
            self.people.append((np.asarray(prose, np.int32), sents))
        self.strings = [p["prose"] for p in b["people"]]      # for the audit
        self.n_people = len(self.people)

    def document_ids(self):
        prose, sents = self.people[self.rng.randrange(self.n_people)]
        ids = prose.tolist()
        if self.augment and sents:
            attrs = list(sents)
            self.rng.shuffle(attrs)
            for a in attrs:
                ids.extend(self.rng.choice(sents[a]))
        return ids

    def document_string(self, b, i):
        """Same construction as document_ids but as text, for the format audit only."""
        p = b["people"][i]
        s = p["prose"]
        if self.augment:
            attrs = list(p["facts"]); random.Random(i).shuffle(attrs)
            s += "".join(" " + render(random.Random(i * 7 + k).choice(self.templates[a]),
                                      p["name"], p["facts"][a]) for k, a in enumerate(attrs))
        return s

    def row(self, seq_len, sep_ids):
        out = []
        while len(out) < seq_len:
            out.extend(self.document_ids())
            out.extend(sep_ids)
        return out[:seq_len]


class Mixer:
    """Rows from B with probability b_ratio, otherwise from the generic pool."""

    def __init__(self, b_renderer, generic_pool, b_ratio, seq_len, sep_ids, seed):
        self.b, self.g, self.ratio = b_renderer, generic_pool, float(b_ratio)
        self.seq_len, self.sep = seq_len, list(sep_ids)
        self.rng = random.Random(seed + 17)
        if self.ratio < 1.0 and (generic_pool is None or not generic_pool.docs):
            raise SystemExit("b_ratio < 1 needs a generic token pool (run.py fetch-generic)")

    def batch(self, n):
        rows = []
        for _ in range(n):
            src = self.b if (self.ratio >= 1.0 or self.rng.random() < self.ratio) else self.g
            rows.append(src.row(self.seq_len, self.sep))
        return np.asarray(rows, dtype=np.int64)
