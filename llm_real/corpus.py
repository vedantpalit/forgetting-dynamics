"""The real-entity B corpus, from a public snapshot, with no API key and no bulk download.

SOURCE. `wikimedia/structured-wikipedia` on the Hugging Face Hub (Wikimedia Enterprise
snapshot, English, 7.6M articles, Parquet, public). Three things make it the right source:
`date_created` (the article's creation date, populated for recent pages), `abstract` (the
lead section as prose), and parsed `infoboxes`. OLMo 2's cutoff is December 2023
(model card), so articles created from 2024-01-01 on are about entities the model did not
see -- and the B-gate (gate_b.py) checks that fact by fact rather than trusting the date.

Counted through the public filter endpoint (2026-09-14): 8,442 English articles created
after 2024-01-01 carry a creation date; 1,721 of them have a person-type infobox.

HOW IT IS FETCHED. The datasets-server `/filter` endpoint accepts a SQL WHERE over the
snapshot's DuckDB index and pages 100 rows at a time, so the biographies come down in ~18
requests without touching the 37 GB of shards. Generic (pre-cutoff) text for the mixing
arm comes from whole Parquet shards, read with pyarrow over HTTP, a couple of shards being
plenty. Both run on a login node with internet; jobs read the files they write.

FACTS. From the infobox, mapped to four attributes -- the same semantic types as A's
relations: birth place (P19-like), alma mater (P69), employer or team (P108/P54), occupation
(P106). Citizenship was dropped: the field mixes demonyms and country names, which no single
template reads correctly. Values are normalized to a short surface form (first
comma-separated item, parentheses and dates stripped) and kept only if they tokenize to a
first token that is not the first token of ANY answer in A (the disjoint-halves condition,
as in llm/build_b.py). A person is kept with the facts that survive; people with no
surviving fact are dropped.

TEXT. The article's abstract, verbatim, is the person's prose. The augmented arm appends
fact sentences rendered from templates in the same encyclopedic register (templates.py);
the strict arm trains on the abstract alone. Format is audited at train time (tokens.py).
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request

DATASET = "wikimedia/structured-wikipedia"
CONFIG = "enwiki_namespace_0"
FILTER = "https://datasets-server.huggingface.co/filter"
SHARD = ("https://huggingface.co/datasets/wikimedia/structured-wikipedia/resolve/main/"
         "enwiki/data/enwiki_namespace_0_{i}.parquet")

PERSON_INFOBOXES = [
    "Infobox person", "Infobox officeholder", "Infobox sportsperson", "Infobox football biography",
    "Infobox scientist", "Infobox academic", "Infobox writer", "Infobox artist",
    "Infobox musical artist", "Infobox politician", "Infobox cricketer",
    "Infobox basketball biography", "Infobox ice hockey biography", "Infobox baseball biography",
    "Infobox gymnast", "Infobox swimmer", "Infobox tennis biography", "Infobox military person",
    "Infobox judge", "Infobox economist", "Infobox engineer", "Infobox chess biography",
    "Infobox racing driver", "Infobox actor", "Infobox comedian", "Infobox model",
    "Infobox rugby biography", "Infobox NFL biography", "Infobox golfer", "Infobox badminton player",
    "Infobox boxer", "Infobox climber", "Infobox architect", "Infobox philosopher",
    "Infobox religious biography", "Infobox Christian leader", "Infobox noble", "Infobox royalty",
    "Infobox criminal", "Infobox skier", "Infobox athlete", "Infobox cyclist", "Infobox MMA fighter",
    "Infobox professional wrestler", "Infobox YouTube personality", "Infobox journalist",
    "Infobox medical person", "Infobox aviator", "Infobox astronaut", "Infobox chef",
    "Infobox designer", "Infobox fashion designer", "Infobox pageant titleholder",
    "Infobox mountaineer", "Infobox handball biography", "Infobox volleyball biography",
    "Infobox figure skater", "Infobox speed skater", "Infobox curler", "Infobox sailor",
    "Infobox rower", "Infobox fencer", "Infobox judoka", "Infobox wrestler", "Infobox weightlifter",
    "Infobox sport shooter", "Infobox archer", "Infobox equestrian", "Infobox darts player",
    "Infobox snooker player", "Infobox esports player", "Infobox go player", "Infobox poker player",
    "Infobox table tennis player", "Infobox squash player", "Infobox surfer", "Infobox skateboarder",
    "Infobox snowboarder", "Infobox biathlete", "Infobox luger", "Infobox bobsledder",
    "Infobox triathlete", "Infobox canoeist", "Infobox diver", "Infobox water polo biography",
    "Infobox field hockey player", "Infobox lacrosse player", "Infobox netball biography",
    "Infobox Australian rules football biography", "Infobox Gaelic games player",
    "Infobox softball player", "Infobox sumo wrestler", "Infobox kickboxer",
    "Infobox Muay Thai fighter", "Infobox karateka", "Infobox taekwondo player",
    "Infobox sport wrestler",
]

# infobox field label (lowercased, punctuation stripped) -> attribute
# Explicit fields are preferred over composite ones: "Place of birth" before "Born" (which is
# "date place" or "full name date place"). Parties and offices are NOT employers; positions,
# fields and "known for" are NOT occupations -- they were tried and produce a different value
# type from the rest of the pool.
FIELD_MAP = {
    "birth_place": ["place of birth", "birth place", "birthplace", "born"],
    "alma_mater": ["alma mater", "education", "educated at"],
    "employer": ["employer", "employers", "current team", "team", "club", "current club",
                 "institutions", "workplaces", "organization", "organisation"],
    "occupation": ["occupation", "occupations", "occupation(s)", "profession"],
}
_DEGREE = re.compile(r"\b(phd|ph\.d|ba|bsc|bs|ma|msc|mba|md|jd|llb|llm|degree|diploma|"
                     r"bachelor|master|doctor|doctorate|certificate)\b", re.I)
_LABEL_TO_ATTR = {lab: attr for attr, labs in FIELD_MAP.items() for lab in labs}


def _get(url, tries=60, wait=8):
    """GET with retries; the filter index reports 'loading' for a while on first use."""
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                d = json.load(r)
            if "rows" in d or "error" not in d:
                return d
            if "invalid" in d.get("error", ""):
                raise SystemExit(d["error"])
        except SystemExit:
            raise
        except Exception:
            pass
        time.sleep(wait)
    raise SystemExit(f"gave up on {url[:120]}")


def fetch_people(min_date="2024-01-01T00:00:00", out_path="llm_real/out/people_raw.json",
                 page=100):
    """All post-cutoff biographies, paged through the public filter endpoint."""
    where = (f"\"date_created\" > '{min_date}' AND ("
             + " OR ".join(f"\"infoboxes\" LIKE '%{n}%'" for n in PERSON_INFOBOXES) + ")")
    rows, off = [], 0
    while True:
        url = (f"{FILTER}?dataset={DATASET}&config={CONFIG}&split=train"
               f"&where={urllib.parse.quote(where)}&offset={off}&length={page}")
        d = _get(url)
        rows += [r["row"] for r in d["rows"]]
        print(f"  fetched {len(rows)} / {d.get('num_rows_total', '?')}", flush=True)
        if len(d["rows"]) < page:
            break
        off += page
    keep = ["name", "identifier", "url", "date_created", "description", "abstract",
            "infoboxes", "main_entity"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump([{k: r.get(k) for k in keep} for r in rows], open(out_path, "w"), ensure_ascii=False)
    print(f"wrote {len(rows)} people -> {out_path}")
    return out_path


def fetch_articles(min_date="2024-01-01T00:00:00", out_path="llm_real/out/b_corpus.json",
                   page=100, min_chars=200, held_frac=0.1, seed=0):
    """ALL post-cutoff articles (any type, no infobox condition): the lead section as a
    document. No facts, no templates -- the real-text arm. Writes {docs, held, meta}: a 90/10
    split by seed, the held-out leads never trained on, both register-checked with the same
    rule as B and the generic pool."""
    from llm_real.tokens import violation
    where = f"\"date_created\" > '{min_date}'"
    rows, off = [], 0
    while True:
        url = (f"{FILTER}?dataset={DATASET}&config={CONFIG}&split=train"
               f"&where={urllib.parse.quote(where)}&offset={off}&length={page}")
        d = _get(url)
        rows += [r["row"] for r in d["rows"]]
        print(f"  fetched {len(rows)} / {d.get('num_rows_total', '?')}", flush=True)
        if len(d["rows"]) < page:
            break
        off += page
    docs, n_short, n_bad = [], 0, 0
    for r in rows:
        ab = (r.get("abstract") or "").strip()
        if len(ab) < min_chars:
            n_short += 1; continue
        if violation(ab):
            n_bad += 1; continue
        docs.append({"name": r.get("name"), "date_created": r.get("date_created"), "text": ab})
    rng = __import__("random").Random(seed)
    rng.shuffle(docs)
    n_held = int(round(held_frac * len(docs)))
    held, train = docs[:n_held], docs[n_held:]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"docs": train, "held": held,
               "meta": {"min_date": min_date, "fetched": len(rows), "short": n_short,
                        "register_violations": n_bad, "min_chars": min_chars, "seed": seed}},
              open(out_path, "w"), ensure_ascii=False)
    print(f"{len(rows)} post-cutoff articles -> {len(docs)} leads >= {min_chars} chars "
          f"({n_short} short, {n_bad} register violations) -> {len(train)} train / {len(held)} held "
          f"-> {out_path}")
    return out_path


def infobox_fields(infoboxes_json):
    """Flatten the parsed infobox into an ordered {label: value} of its leaf fields."""
    out = {}
    try:
        ib = json.loads(infoboxes_json) if infoboxes_json else []
    except Exception:
        return out

    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "field" and n.get("name") and n.get("value") is not None:
                out.setdefault(n["name"], n["value"])
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for x in n:
                walk(x)
    walk(ib)
    return out


_DATE = re.compile(r"\b(\d{1,2}\s+)?(January|February|March|April|May|June|July|August|September|"
                   r"October|November|December)\s+\d{1,2},?\s*\d{4}|\b\d{4}-\d{2}-\d{2}\b|"
                   r"\b\d{4}\b|\(age\s*\d+\)|\(aged\s*\d+\)", re.I)


def normalize_value(attr, raw, name=""):
    """A short surface form: dates and parentheticals removed, first list item, cleaned.
    For "Born" the place is whatever FOLLOWS the last date, since the field reads
    "[full name] date place"."""
    s = str(raw)
    s = re.sub(r"\[.*?\]", " ", s)
    s = re.sub(r"\(.*?\)", " ", s)
    if attr == "birth_place":
        m = None
        for m in _DATE.finditer(s):
            pass
        if m is not None:
            s = s[m.end():]
    s = _DATE.sub(" ", s)
    s = s.replace("\n", ", ")
    parts = [p.strip(" ,.;") for p in re.split(r",|;|•|\|| and | or ", s) if p.strip(" ,.;")]
    if not parts:
        return None
    v = re.sub(r"\s+", " ", parts[0]).strip()
    if attr == "occupation":
        v = v.lower()
    if attr == "alma_mater" and (_DEGREE.search(v) or re.fullmatch(r"[A-Za-z]{1,3}(\.[A-Za-z]{1,3})*\.?", v)):
        return None
    if attr == "birth_place" and name and v.lower() in name.lower():
        return None
    if len(v) < 3 or len(v) > 40 or re.search(r"\d", v) or v.lower() in ("or", "and", "unknown"):
        return None
    return v


def person_facts(row):
    """{attr: value} for one raw row, best field per attribute, normalized. Fields are
    visited in FIELD_MAP's preference order, not the infobox's."""
    fields = infobox_fields(row.get("infoboxes"))
    norm = {re.sub(r"[^a-z() ]", "", k.lower()).strip(): v for k, v in fields.items()}
    facts = {}
    ordered = [(lab, norm[lab]) for attr in FIELD_MAP for lab in FIELD_MAP[attr] if lab in norm]
    for label, value in ordered:
        attr = _LABEL_TO_ATTR.get(label)
        if attr is None or attr in facts:
            continue
        v = normalize_value(attr, value, row.get("name", ""))
        if v:
            facts[attr] = v
    return facts


def fetch_generic(n_shards=2, out_path="llm_real/out/generic_tokens.npy", model_id=None,
                  min_chars=400, max_docs=60000, seed=0):
    """Pre-cutoff articles (no creation date, i.e. old pages) from whole shards, as one
    pre-tokenized document pool. Needs pyarrow and the tokenizer; login node only.

    Streams each shard by row group with only the two columns it needs: a whole shard
    decompressed at once (sections, references, tables) is several GB and got the process
    killed on a login node. Leads only -- the abstract is the document."""
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id or "allenai/OLMo-2-0425-1B")
    rng = __import__("random").Random(seed)
    docs = []
    for i in range(n_shards):
        url = SHARD.format(i=i)
        local = f"/tmp/enwiki_shard_{i}.parquet"
        if not os.path.exists(local):
            print(f"  downloading shard {i} ...", flush=True)
            urllib.request.urlretrieve(url, local)
        from llm_real.tokens import violation      # the same register rule B is filtered by
        pf = pq.ParquetFile(local)
        n_bad = 0
        for batch in pf.iter_batches(batch_size=2000, columns=["abstract", "date_created"]):
            for ab, dc in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                if dc is None and ab and len(ab) >= min_chars:
                    if violation(ab):
                        n_bad += 1
                        continue
                    docs.append(ab.strip())
            if len(docs) >= max_docs:
                break
        print(f"  shard {i}: {len(docs)} documents so far ({n_bad} dropped by the format rule)", flush=True)
        if len(docs) >= max_docs:
            break
    rng.shuffle(docs)
    docs = docs[:max_docs]
    ids = [np.asarray(tok(d, add_special_tokens=False).input_ids, dtype=np.int32) for d in docs]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, np.array(ids, dtype=object), allow_pickle=True)
    print(f"wrote {len(ids)} generic documents, {sum(len(x) for x in ids):,} tokens -> {out_path}")
    return out_path
