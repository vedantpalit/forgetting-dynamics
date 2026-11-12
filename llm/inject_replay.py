"""Replay test in OLMo 2 1B: inject B exactly as llm/inject.py does, and ADD a little replay of A.

THE QUESTION. If a small subset R of the old facts A is replayed during the injection of B, what
happens to the old facts that are NEVER replayed (H = A minus R)? Nothing from the pretrained
weights is used: the only added ingredient is text.

WHAT IS THE SAME AS llm/inject.py (imported, not copied): the model and dtype handling, build_a /
build_b, DocSampler for B's documents, score, summarise, eval_schedule. The training loop below is
inject.py's loop copied line for line (AdamW betas (0.9, 0.95), wd 0, linear warmup, fp32 master
weights with bf16 autocast, clip 1.0, accumulation over micro-batches). The only changes are the
ones marked REPLAY.

THE SPLIT (fixed, independent of the training seed). Subjects of A are sorted and a fraction
--replay_frac of them is drawn with random.Random(--split_seed). R = every gated fact (copy and
noncopy) of a drawn subject; H = all other facts. No subject has facts on both sides.

REPLAY TEXT. One sentence per replayed fact: the CounterFact prompt the evaluation feeds for that
fact, then " " + target.strip() -- the string whose first token is the scored token (evalsets.
first_token_id). No full stop is added; nothing else is added. It is checked at start-up, for every
fact of R, that the sentence tokenises as [prompt tokens][scored first token]..., and that appending
the separator does not change those tokens. A failure stops the run.

PACKING. As DocSampler.batch: sentences are drawn uniformly with replacement, each followed by
"\n\n", tokenised with add_special_tokens=False (no BOS, no EOS), concatenated until seq_len is
reached and truncated to seq_len. labels = input_ids, so every position is in the loss, as for B.

THE STEP. The 64 B sequences of a step are drawn from the same DocSampler(seed) in the same order as
in inject.py; the replay sampler has its own random.Random and never touches DocSampler's, so B's
token stream is identical with and without replay for a given seed. Replay sequences are ADDED
(n_rep per step while replay_from <= step <= replay_to). The loss of the step is the mean token
cross-entropy over ALL n_total = 64 + n_rep sequences, every sequence weighted equally: each
micro-batch of m sequences contributes loss_mean * m / n_total. With n_rep = 0 this is
loss / accum, i.e. inject.py exactly.

--replay_template other (the default "eval" is the behaviour above).
Each replayed fact is stated in ONE declarative sentence per relation (OTHER_TEMPLATES) that is not a
CounterFact prompt template: the evaluation prompt of the replayed facts is then never trained on.
Checked at start-up, and a failure stops the run: every sentence contains its subject and its answer;
the answer is tokenised to the scored first token; no evaluation prompt of ANY old fact (R or H) is a
substring of any replay sentence; and, per relation, the three words before the answer differ from the
last three words of every evaluation template that relation has among the old facts.

Run:
  .venv-llm/bin/python -m llm.inject_replay --lr 1e-5 --steps 400 --facts llm/out/b_facts_small.json \
      --cond spread --replay_seqs 2 --replay_from 1 --replay_to 400 --seed 0
"""
import argparse
import json
import math
import os
import random
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.evalsets import build_a, build_b
from llm.inject import DATASET_ID, MODEL_ID, DocSampler, eval_schedule, score, summarise

SEPARATOR = "\n\n"                      # DocSampler.batch's document separator
REPLAY_RNG_OFFSET = 1_000_003           # replay sampler seed = training seed + this


def split_replay(a_items, frac, seed):
    """R/H split by SUBJECT. Returns (sorted indices of R, sorted indices of H, R's subjects)."""
    subjects = sorted({it["subject"] for it in a_items})
    k = int(round(frac * len(subjects)))
    r_subjects = set(random.Random(seed).sample(subjects, k))
    r_idx = [i for i, it in enumerate(a_items) if it["subject"] in r_subjects]
    h_idx = [i for i, it in enumerate(a_items) if it["subject"] not in r_subjects]
    return r_idx, h_idx, sorted(r_subjects)


def replay_sentence(it):
    """The eval prompt followed by the answer exactly as it is scored (leading space)."""
    return it["prompt"] + " " + it["target"].strip()


def check_sentences(tok, a_items, r_idx):
    """Every replay sentence must tokenise as [prompt][scored first token]... with or without the
    separator behind it. Returns the bare token length of each sentence; raises on any mismatch."""
    bad, lengths = [], []
    for i in r_idx:
        it = a_items[i]
        s = replay_sentence(it)
        p_ids = tok(it["prompt"], add_special_tokens=False).input_ids
        bare = tok(s, add_special_tokens=False).input_ids
        full = tok(s + SEPARATOR, add_special_tokens=False).input_ids
        ok = (bare[:len(p_ids)] == p_ids and len(bare) > len(p_ids)
              and bare[len(p_ids)] == it["first_token"]
              and full[:len(bare)] == bare and len(full) > len(bare))
        if not ok:
            bad.append((i, s))
        lengths.append(len(bare))
    if bad:
        raise SystemExit(f"replay sentence tokenisation mismatch for {len(bad)} facts, "
                         f"e.g. {bad[:5]!r}")
    return lengths


# --replay_template other: ONE declarative sentence per relation, never a CounterFact template.
# "repo" = copied verbatim from llm/build_b_entityq.py RELATIONS[<relation>][2][0] (the first
# phrasing listed there for that Wikidata relation), with {name}->{subject}, {v}->{answer}.
# "written" = written for this script. P19 has a repo entry too, but it is btemplates' birth_city
# list, i.e. the frames B itself is trained on in this run, so it is not used.
OTHER_TEMPLATES = {
    "P176": ("{subject} came off the production line of {answer}.", "repo"),
    "P159": ("{subject} ran its operations from {answer}.", "repo"),
    "P407": ("{subject} was composed in {answer}.", "repo"),
    "P178": ("The company behind {subject} is {answer}.", "written"),
    "P449": ("The network that first broadcast {subject} was {answer}.", "written"),
    "P27": ("The state that counts {subject} among its citizens is {answer}.", "written"),
    "P108": ("{subject} has a job with {answer}.", "written"),
    "P36": ("As its capital, {subject} has the city of {answer}.", "written"),
    "P364": ("{subject} was originally produced in the {answer} language.", "written"),
    "P140": ("In matters of faith, {subject} observes {answer}.", "written"),
    "P131": ("The area that contains {subject} is {answer}.", "written"),
    "P138": ("{subject} takes its name from {answer}.", "written"),
    "P103": ("{subject} learned to speak in {answer}.", "written"),
    "P127": ("The company that owns {subject} is {answer}.", "written"),
    "P1303": ("The instrument of {subject} is the {answer}.", "written"),
    "P276": ("The site of {subject} is {answer}.", "written"),
    "P37": ("{subject} uses {answer} as its official language.", "written"),
    "P19": ("{subject} came into the world in {answer}.", "written"),
    "P30": ("Among the continents, {subject} is part of {answer}.", "written"),
}


def other_sentence(it):
    """The fact stated in its relation's OTHER_TEMPLATES sentence."""
    return OTHER_TEMPLATES[it["relation_id"]][0].format(subject=it["subject"].strip(),
                                                        answer=it["target"].strip())


def _last3(frame):
    """Last three words of a frame; the subject slot '{}' counts as one word."""
    return tuple(re.findall(r"\{\}|[a-z]+", frame.lower())[-3:])


def check_other(tok, a_items, r_idx, dataset):
    """Start-up checks for --replay_template other. Returns (token lengths, report); raises on
    any failure."""
    # the evaluation templates the old facts actually use, per relation (as build_a picks the
    # prompt: first dataset row of each (subject, relation))
    tmpl_by = {}
    for r in dataset:
        tmpl_by.setdefault((r["subject"], r["relation_id"]), r["relation"])
    eval_frames = {}
    for it in a_items:
        eval_frames.setdefault(it["relation_id"], set()).add(tmpl_by[(it["subject"], it["relation_id"])])
    all_prompts = sorted({it["prompt"] for it in a_items})
    bad, lengths = [], []
    rels = sorted({a_items[i]["relation_id"] for i in r_idx})
    for rid in rels:                                   # frame check, once per relation
        mine = _last3(OTHER_TEMPLATES[rid][0].split("{answer}")[0].replace("{subject}", "{}"))
        clash = [f for f in eval_frames[rid] if _last3(f) == mine]
        if clash:
            bad.append(("last three words equal an evaluation template", rid, clash))
    for i in r_idx:
        it = a_items[i]
        s = other_sentence(it)
        subj, ans = it["subject"].strip(), it["target"].strip()
        if subj not in s or ans not in s:
            bad.append(("subject or answer missing", i, s))
        hit = [p for p in all_prompts if p in s]
        if hit:
            bad.append(("an evaluation prompt is a substring", i, s, hit[:2]))
        before = OTHER_TEMPLATES[it["relation_id"]][0].split("{answer}")[0].format(subject=subj).rstrip()
        pre_ids = tok(before, add_special_tokens=False).input_ids
        ids = tok(s, add_special_tokens=False).input_ids
        full = tok(s + SEPARATOR, add_special_tokens=False).input_ids
        if not (ids[:len(pre_ids)] == pre_ids and len(ids) > len(pre_ids)
                and ids[len(pre_ids)] == it["first_token"]
                and full[:len(pre_ids) + 1] == ids[:len(pre_ids) + 1]):
            bad.append(("answer not tokenised to the scored first token", i, s))
        lengths.append(len(full))                      # sentence AND separator must fit to count
    if bad:
        raise SystemExit(f"--replay_template other: {len(bad)} check failures, e.g. {bad[:5]!r}")
    report = {"n_sentences": len(r_idx), "relations": rels,
              "n_eval_prompts_checked_as_substring": len(all_prompts),
              "eval_templates_per_relation": {rid: sorted(eval_frames[rid]) for rid in rels},
              "all_checks_passed": True}
    return lengths, report


class ReplaySampler:
    """Packs replay sentences the way DocSampler.batch packs documents. Own RNG."""

    def __init__(self, sentences, lengths, tok, seq_len, seed):
        self.sentences, self.lengths = sentences, lengths
        self.tok, self.seq_len = tok, seq_len
        self.rng = random.Random(seed)
        self.shown = [0] * len(sentences)      # complete showings (answer inside the sequence)
        self.n_shown = 0

    def batch(self, batch_size):
        rows = []
        for _ in range(batch_size):
            ids = []
            while len(ids) < self.seq_len:
                k = self.rng.randrange(len(self.sentences))
                # counted only if the whole sentence survives the truncation to seq_len
                if len(ids) + self.lengths[k] <= self.seq_len:
                    self.shown[k] += 1
                    self.n_shown += 1
                ids.extend(self.tok(self.sentences[k] + SEPARATOR,
                                    add_special_tokens=False).input_ids)
            rows.append(ids[:self.seq_len])
        return torch.tensor(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=4, help="micro-batch; memory only")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--tokens_per_step", type=int, default=32768,
                    help="B tokens per optimizer step, reached by accumulation (replay is extra)")
    ap.add_argument("--warmup", type=int, default=100, help="linear LR warmup steps")
    ap.add_argument("--eval_batch", type=int, default=128)
    ap.add_argument("--gate", default="llm/out/gate.json")
    ap.add_argument("--facts", default="llm/out/b_facts_small.json")
    ap.add_argument("--model_id", default=MODEL_ID,
                    help="HF id; the gate (--gate) must have been built on the same model")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    # REPLAY
    ap.add_argument("--cond", default="none", help="label only; goes into the output name")
    ap.add_argument("--replay_seqs", type=int, default=0,
                    help="replay sequences ADDED to each step inside the replay window")
    ap.add_argument("--replay_from", type=int, default=1, help="first step with replay (inclusive)")
    ap.add_argument("--replay_to", type=int, default=10 ** 9, help="last step with replay (inclusive)")
    ap.add_argument("--replay_frac", type=float, default=0.125, help="fraction of A's SUBJECTS in R")
    ap.add_argument("--split_seed", type=int, default=0, help="R/H split seed, not the training seed")
    ap.add_argument("--replay_template", choices=["eval", "other"], default="eval",
                    help="eval: the evaluation prompt + answer (default); "
                         "other: one declarative non-CounterFact sentence per relation")
    ap.add_argument("--extra_evals", default="85,90,110,120,140,160",
                    help="evaluation steps added to inject.py's grid")
    a = ap.parse_args()
    out_path = a.out or f"llm/out/inject_replay_small_{a.cond}_seed{a.seed}.json"
    t_start = time.time()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  lr={a.lr:g}  steps={a.steps}  batch={a.batch_size}x{a.seq_len}")
    print(f"REPLAY cond={a.cond}  replay_seqs={a.replay_seqs}  window=[{a.replay_from}, "
          f"{a.replay_to}]  replay_frac={a.replay_frac}  split_seed={a.split_seed}")
    if a.replay_template != "eval":
        print(f"REPLAY template mode: {a.replay_template}")

    tok = AutoTokenizer.from_pretrained(a.model_id)
    # fp32 MASTER WEIGHTS, bf16 only for the matmuls (see llm/inject.py).
    model = AutoModelForCausalLM.from_pretrained(a.model_id, dtype=torch.float32).to(device)

    ds = load_dataset(DATASET_ID, split="train")
    a_items, a_pools, a_info = build_a(tok, a.gate, ds)
    b_items, b_pools, facts = build_b(tok, a.facts)
    # as llm/inject.py
    TYPE_OF = {"employer": ("P176", "P178", "P108"), "maker": ("P176", "P178", "P108"),
               "birth_city": ("P159", "P36"), "citizenship": ("P27",), "language": ("P103", "P364"),
               "country": ("P27",)}       # EntityQuestions P17: places -> countries
    comp = {rid for at in facts["attrs"] for rid in TYPE_OF.get(at, ())}
    for it in a_items:
        it["compete"] = "same_type" if it["relation_id"] in comp else "other_type"
    print(f"A facts of B's answer types (compete=same_type): {sum(it['compete'] == 'same_type' for it in a_items)} "
          f"of {len(a_items)}  (relations {sorted(comp)})")
    print(f"A: {a_info['n_A']} facts, rank-eligible {a_info['rank_eligible']} "
          f"({a_info['rank_coverage']:.1%})")
    print(f"   copy {a_info['copy']}  |  strata {a_info['strata']}  |  {a_info['touched']}")
    print(f"B: {len(b_items)} eval items on held-out phrasings over {len(b_pools)} attributes")

    # REPLAY: the split, the sentences, the sampler ------------------------------------------
    r_idx, h_idx, r_subjects = split_replay(a_items, a.replay_frac, a.split_seed)
    r_set = set(r_idx)
    for i, it in enumerate(a_items):
        it["split"] = "R" if i in r_set else "H"
        it["split_copy"] = f"{it['split']}_{it['copy']}"
    assert not ({a_items[i]["subject"] for i in r_idx} & {a_items[i]["subject"] for i in h_idx})
    n_subj = len({it["subject"] for it in a_items})
    print(f"REPLAY split: {len(r_subjects)} of {n_subj} subjects -> R {len(r_idx)} facts "
          f"({sum(a_items[i]['copy'] == 'noncopy' for i in r_idx)} noncopy), "
          f"H {len(h_idx)} facts ({sum(a_items[i]['copy'] == 'noncopy' for i in h_idx)} noncopy)")
    other_report = None
    if a.replay_template == "other":
        r_sentences = [other_sentence(a_items[i]) for i in r_idx]
        r_lengths, other_report = check_other(tok, a_items, r_idx, ds)
        for rid in other_report["relations"]:
            print(f"   OTHER template {rid} [{OTHER_TEMPLATES[rid][1]}]: {OTHER_TEMPLATES[rid][0]!r}")
    else:
        r_sentences = [replay_sentence(a_items[i]) for i in r_idx]
        r_lengths = check_sentences(tok, a_items, r_idx)
    print(f"REPLAY sentences: {len(r_sentences)}, bare length {min(r_lengths)}-{max(r_lengths)} "
          f"tokens (mean {sum(r_lengths) / len(r_lengths):.2f}); tokenisation check passed")
    for s in r_sentences[:2]:
        print(f"   example sentence: {s!r}")
    demo = ReplaySampler(r_sentences, r_lengths, tok, a.seq_len, 12345).batch(1)[0].tolist()
    print(f"   example sequence ({len(demo)} tokens, first id {demo[0]}, bos id {tok.bos_token_id}, "
          f"eos id {tok.eos_token_id}); decoded prefix: {tok.decode(demo[:60])!r}")
    replay = ReplaySampler(r_sentences, r_lengths, tok, a.seq_len, a.seed + REPLAY_RNG_OFFSET)

    sampler = DocSampler(facts, tok, a.seq_len, a.seed)
    print(f"documents rendered fresh each step from {len(facts['people'])} people x "
          f"{len(facts['attrs'])} attributes; {a.batch_size * a.seq_len:,} tokens/step")

    micro_tokens = a.batch_size * a.seq_len
    accum = max(a.tokens_per_step // micro_tokens, 1)
    print(f"  {micro_tokens:,} tokens/micro-batch x {accum} accumulated = "
          f"{micro_tokens * accum:,} tokens/step; {a.warmup}-step linear warmup")
    n_b_seqs = accum * a.batch_size

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    extra = [int(s) for s in a.extra_evals.split(",") if s.strip()]
    schedule = sorted(set(eval_schedule(a.steps)) | {s for s in extra if s <= a.steps})
    print(f"eval steps: {schedule}")
    curve = []
    replay_seqs_cum = 0
    replay_seqs_per_step, replay_sentences_per_step = [], []
    loss_b_per_step, loss_replay_per_step = [], []

    def evaluate(step):
        aa, ar = score(model, tok, a_items, a_pools, device, a.eval_batch)
        ba, br = score(model, tok, b_items, b_pools, device, a.eval_batch)
        rec = {"step": step}
        rec.update(summarise(aa, ar, a_items, "A", "copy"))
        rec.update(summarise(aa, ar, a_items, "Arel", "stratum"))
        rec.update(summarise(aa, ar, a_items, "Atouch", "touched"))
        rec.update(summarise(aa, ar, a_items, "Acomp", "compete"))
        rec.update(summarise(aa, ar, a_items, "Asplit", "split_copy"))          # REPLAY
        rec.update(summarise(ba, br, b_items, "B", "stratum"))
        # per-fact A outcomes, so any stratum can be read offline (1106 ints + floats per eval)
        rec["A/correct"] = [int(v) for v in aa]
        rec["A/rank_each"] = [None if math.isnan(v) else round(float(v), 2) for v in ar]
        # REPLAY exposure so far
        rec["replay/seqs_cum"] = replay_seqs_cum
        rec["replay/sentences_cum"] = replay.n_shown
        rec["replay/mean_shows_per_fact"] = replay.n_shown / len(r_sentences)
        curve.append(rec)
        print(f"  [step {step:>4}] A noncopy={rec['A/noncopy/acc']:.4f}"
              f"+-{rec['A/noncopy/acc_se']:.4f} rank={rec.get('A/noncopy/rank', float('nan')):.3f}"
              f" | copy={rec['A/copy/acc']:.4f} all={rec['A/ALL/acc']:.4f}"
              f" | b_typed={rec['Atouch/b_typed/acc']:.4f} "
              f"untouched={rec['Atouch/untouched/acc']:.4f}"
              f" | B={rec['B/ALL/acc']:.4f}"
              f" | H noncopy={rec['Asplit/H_noncopy/acc']:.4f} "
              f"R noncopy={rec['Asplit/R_noncopy/acc']:.4f} "
              f"shows/fact={rec['replay/mean_shows_per_fact']:.2f}")
        model.train()

    evaluate(0)
    for step in range(1, a.steps + 1):
        # Warmup, as llm/inject.py.
        for gp in opt.param_groups:
            gp["lr"] = a.lr * min(step / max(a.warmup, 1), 1.0)
        # REPLAY: the step holds n_b_seqs B sequences plus n_rep replay sequences; the loss is the
        # mean over all of them. With n_rep == 0, a.batch_size / n_total == 1 / accum.
        n_rep = a.replay_seqs if a.replay_from <= step <= a.replay_to else 0
        n_total = n_b_seqs + n_rep
        total = 0.0
        for _ in range(accum):
            batch = sampler.batch(a.batch_size).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=batch, labels=batch).loss * (a.batch_size / n_total)
            loss.backward()
            total += float(loss.item())
        # REPLAY micro-batches, same micro-batch size, added on top
        rep_total, left, shown_before = 0.0, n_rep, replay.n_shown
        while left > 0:
            m = min(a.batch_size, left)
            batch = replay.batch(m).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=batch, labels=batch).loss * (m / n_total)
            loss.backward()
            rep_total += float(loss.item())
            left -= m
        replay_seqs_cum += n_rep
        replay_seqs_per_step.append(n_rep)
        replay_sentences_per_step.append(replay.n_shown - shown_before)
        loss_b_per_step.append(total * n_total / n_b_seqs)                 # mean CE on B sequences
        loss_replay_per_step.append(rep_total * n_total / n_rep if n_rep else None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step in schedule:
            rl = loss_replay_per_step[-1]
            print(f"    train loss {total + rep_total:.4f}  (lr {opt.param_groups[0]['lr']:.3g})"
                  f"  B {loss_b_per_step[-1]:.4f}  replay "
                  f"{'-' if rl is None else format(rl, '.4f')}  n_rep {n_rep}")
            evaluate(step)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"model": a.model_id, "facts": a.facts, "gate": a.gate, "lr": a.lr,
                   "steps": a.steps, "seed": a.seed, "batch_size": a.batch_size,
                   "a_relation": [it["relation_id"] for it in a_items],
                   "a_copy": [it["copy"] for it in a_items],
                   "a_compete": [it["compete"] for it in a_items],
                   "seq_len": a.seq_len, "tokens_per_step": micro_tokens * accum,
                   "warmup": a.warmup, "a_info": a_info,
                   # REPLAY
                   "a_subject": [it["subject"] for it in a_items],
                   "a_prompt": [it["prompt"] for it in a_items],
                   "a_target": [it["target"] for it in a_items],
                   "a_first_token": [it["first_token"] for it in a_items],
                   "a_split": [it["split"] for it in a_items],
                   "replay": {
                       "cond": a.cond, "replay_seqs": a.replay_seqs,
                       "replay_from": a.replay_from, "replay_to": a.replay_to,
                       "replay_frac": a.replay_frac, "split_seed": a.split_seed,
                       "replay_template": a.replay_template,
                       "other_templates": (None if other_report is None else
                                           {rid: list(OTHER_TEMPLATES[rid])
                                            for rid in other_report["relations"]}),
                       "other_checks": other_report,
                       "replay_rng_seed": a.seed + REPLAY_RNG_OFFSET,
                       "n_b_seqs_per_step": n_b_seqs, "separator": SEPARATOR,
                       "loss_average": "mean token CE over all n_b_seqs + n_rep sequences of the step",
                       "n_subjects": n_subj, "r_subjects": r_subjects,
                       "r_idx": r_idx, "h_idx": h_idx,
                       "r_sentences": r_sentences, "r_sentence_tokens": r_lengths,
                       "r_shown": replay.shown,
                       "replay_seqs_per_step": replay_seqs_per_step,
                       "replay_sentences_per_step": replay_sentences_per_step,
                       "replay_seqs_total": replay_seqs_cum,
                       "replay_sentences_total": replay.n_shown,
                       "mean_shows_per_fact": replay.n_shown / len(r_sentences),
                   },
                   "loss_b_per_step": loss_b_per_step,
                   "loss_replay_per_step": loss_replay_per_step,
                   "eval_steps": schedule,
                   "wall_seconds": time.time() - t_start,
                   "curve": curve}, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
