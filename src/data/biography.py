"""Synthetic biography dataset (Zucchet et al. 2025), adapted for continual learning.

Each individual has 6 attributes (birthdate, birthplace, university, major,
company, work_location) rendered through natural-language templates. Sequences
start with a fixed-length name prompt: [BOS, name..., PAD...] followed by the
6 attribute sentences in random order. The prompt region is what conditions
autoregressive generation.

Token role mask (aligned to targets after shifting):
    0      prompt-region padding (untrained)
    7      real non-attribute token (names, template words)
    +k     first token of attribute k-1's value (k in 1..6)
    -k     continuation token of attribute k-1's value
    8      trailing padding after the 6th sentence: trained (the model learns
           to emit PAD as end-of-document and to keep emitting it), but
           excluded from displayed metrics
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.data.base import BaseDataset
from src.data.biography_corpus import NUM_TEMPLATES_PER_ATTRIBUTE
from src.data.metrics import compute_cross_entropy

NUM_ATTRIBUTES = 6


class BiographyPopulation:
    """Token pools and per-person facts, shared across dataset subsets."""

    def __init__(
        self,
        data_path: str,
        num_people: int,
        seed: int = 42,
        support_size: int = 1,
        num_train_templates: int = 20,
        use_french: bool = False,
    ):
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Preprocessed biography data not found at {data_path}. "
                f"Run: uv run python -m src.data.preprocess_biography"
            )
        data = np.load(data_path, allow_pickle=True)

        self.num_people = num_people
        # English tokens are a prefix of the vocab; monolingual runs use only
        # that block so old (pre-French) checkpoint shapes stay compatible.
        self.vocab_size = int(data["vocab_size"]) if use_french or "english_vocab_size" not in data \
            else int(data["english_vocab_size"])
        self.pad_id = int(data["pad_id"])
        self.bos_id = int(data["bos_id"])
        self.x_id = int(data["x_id"])
        self.y_id = int(data["y_id"])
        self.max_name_len = int(data["max_name_len"])
        self.prompt_len = 1 + self.max_name_len  # BOS + padded name
        self.use_french = use_french
        max_sentence_len = int(data["max_sentence_len"])
        if use_french:
            if "max_sentence_len_fr" not in data:
                raise ValueError("No French templates in npz; rerun src.data.preprocess_biography")
            max_sentence_len = max(max_sentence_len, int(data["max_sentence_len_fr"]))
        self.seq_len = self.prompt_len + NUM_ATTRIBUTES * max_sentence_len
        self.num_values_per_attr = data["num_values_per_attr"]

        self.attr_value_ids = [data[f"attr_{k}_value_token_ids"] for k in range(NUM_ATTRIBUTES)]
        self.attr_value_lengths = [data[f"attr_{k}_value_token_lengths"] for k in range(NUM_ATTRIBUTES)]
        self.attr_first_token_ids = [data[f"attr_{k}_first_token_ids"] for k in range(NUM_ATTRIBUTES)]
        langs = ("en", "fr") if use_french else ("en",)
        suffix = {"en": "", "fr": "_fr"}
        self.attr_template_ids = {
            lang: [data[f"attr_{k}_template{suffix[lang]}_token_ids"] for k in range(NUM_ATTRIBUTES)]
            for lang in langs}
        self.attr_template_lengths = {
            lang: [data[f"attr_{k}_template{suffix[lang]}_token_lengths"] for k in range(NUM_ATTRIBUTES)]
            for lang in langs}
        self.name_ids = {p: data[f"{p}_name_token_ids"] for p in ["first", "middle", "last"]}
        self.name_lengths = {p: data[f"{p}_name_token_lengths"] for p in ["first", "middle", "last"]}

        rng = np.random.default_rng(seed)
        self._assign_names(rng)
        self._assign_values(rng, support_size)
        self._split_templates(rng, num_train_templates)

    def _assign_names(self, rng):
        """Assign a unique (first, middle, last) triple to each person."""
        pools = [len(self.name_ids[p]) for p in ["first", "middle", "last"]]
        assert np.prod(pools) >= self.num_people, "Not enough name combinations"
        combos = set()
        triples = []
        while len(combos) < self.num_people:
            triple = tuple(rng.integers(0, n) for n in pools)
            if triple not in combos:
                combos.add(triple)
                triples.append(triple)
        self.person_names = np.array(triples, dtype=np.int32)  # (N, 3)

    def _assign_values(self, rng, support_size):
        """Uniform distribution over a random support of values per (person, attribute)."""
        self.value_probs = []  # list of (N, n_values_k)
        for k in range(NUM_ATTRIBUTES):
            n_values = int(self.num_values_per_attr[k])
            s = min(support_size, n_values)
            noise = rng.gumbel(size=(self.num_people, n_values))
            top_k = np.argsort(-noise, axis=-1)[:, :s]
            # float64: Generator.choice() rejects probs whose float32 sum drifts from 1
            probs = np.zeros((self.num_people, n_values), dtype=np.float64)
            np.put_along_axis(probs, top_k, 1.0 / s, axis=-1)
            self.value_probs.append(probs)

    def _split_templates(self, rng, num_train_templates):
        """Per-(person, attribute) train/eval template split."""
        n_eval = NUM_TEMPLATES_PER_ATTRIBUTE - num_train_templates
        self.train_templates = np.zeros((self.num_people, NUM_ATTRIBUTES, num_train_templates), dtype=np.int32)
        self.eval_templates = np.zeros((self.num_people, NUM_ATTRIBUTES, n_eval), dtype=np.int32)
        for i in range(self.num_people):
            for k in range(NUM_ATTRIBUTES):
                perm = rng.permutation(NUM_TEMPLATES_PER_ATTRIBUTE)
                self.train_templates[i, k] = perm[:num_train_templates]
                self.eval_templates[i, k] = perm[num_train_templates:]

    def name_tokens(self, person: int) -> np.ndarray:
        """Concatenated (first, middle, last) name tokens for a person."""
        parts = []
        for pool, idx in zip(["first", "middle", "last"], self.person_names[person]):
            parts.append(self.name_ids[pool][idx, : int(self.name_lengths[pool][idx])])
        return np.concatenate(parts)


class BiographyDataset(BaseDataset):
    """A subset of people from a shared population."""

    CACHE_MULTIPLIER = 64

    def __init__(self, population: BiographyPopulation, person_ids: np.ndarray, name: str,
                 seed: int = 42, max_eval_people: int = 0, lang_mix: float = 0.0,
                 lang_switch: float = 0.0):
        super().__init__(name=name)
        self.pop = population
        assert (lang_mix == 0.0 and lang_switch == 0.0) or population.use_french, \
            "lang_mix/lang_switch > 0 needs use_french population"
        self.lang_mix = lang_mix  # probability the FIRST sentence is French
        self.lang_switch = lang_switch  # per-sentence probability of switching language
        self.person_ids = np.asarray(person_ids, dtype=np.int32)
        self.vocab_size = population.vocab_size
        self.seq_len = population.seq_len
        self.prompt_len = population.prompt_len
        self._np_rng = np.random.default_rng(seed)
        self._cache = None
        self._cache_idx = 0
        # Per-person training exposures, indexed by global person id. Uniform sampling
        # still gives Binomial spread, and exposure confounds any relatedness result
        # unless it is logged and stratified. Counted at serve time, not generation time.
        self.exposure_counts = np.zeros(population.num_people, dtype=np.int64)

        # Pre-generated eval set: one biography per person, held-out templates;
        # optionally capped to a fixed random subset of people (large populations)
        eval_rng = np.random.default_rng(seed + 200)
        self._eval_ids = self.person_ids
        if 0 < max_eval_people < len(self.person_ids):
            self._eval_ids = eval_rng.choice(self.person_ids, max_eval_people, replace=False)
        seqs, masks, _, self._eval_langs = self._generate(
            len(self._eval_ids), eval_templates=True, rng=eval_rng, one_per_person=True)
        self.eval_inputs = seqs[:, :-1]
        self.eval_targets = seqs[:, 1:]
        self.eval_mask = masks[:, 1:]

    def _generate(self, n, eval_templates, rng, one_per_person=False):
        """Generate n biographies. Returns (sequences, role masks, person ids, langs).

        langs[i] = 1 if biography i STARTS with French templates (drawn with
        prob lang_mix). With lang_switch == 0 that language is shared by all 6
        sentences; otherwise each subsequent sentence switches language with
        prob lang_switch (Markov chain; 0.5 = iid, 1 = strict alternation).
        """
        pop = self.pop
        seqs = np.full((n, self.seq_len), pop.pad_id, dtype=np.int32)
        masks = np.zeros((n, self.seq_len), dtype=np.int8)
        persons = np.zeros(n, dtype=np.int32)
        langs = (rng.random(n) < self.lang_mix).astype(np.int8)

        for i in range(n):
            cur_fr = int(langs[i])
            p = self._eval_ids[i % len(self._eval_ids)] if one_per_person \
                else self.person_ids[rng.integers(len(self.person_ids))]
            persons[i] = p
            name_tokens = pop.name_tokens(p)

            # Prompt region: BOS + name, pad-filled up to prompt_len
            seqs[i, 0] = pop.bos_id
            masks[i, 0] = 7
            seqs[i, 1 : 1 + len(name_tokens)] = name_tokens
            masks[i, 1 : 1 + len(name_tokens)] = 7
            pos = self.prompt_len

            for k in rng.permutation(NUM_ATTRIBUTES):
                lang = "fr" if cur_fr else "en"
                pool = pop.eval_templates[p, k] if eval_templates else pop.train_templates[p, k]
                tmpl_idx = pool[rng.integers(len(pool))]
                tmpl = pop.attr_template_ids[lang][k][tmpl_idx, : int(pop.attr_template_lengths[lang][k][tmpl_idx])]
                val_idx = rng.choice(pop.value_probs[k].shape[1], p=pop.value_probs[k][p])
                val = pop.attr_value_ids[k][val_idx, : int(pop.attr_value_lengths[k][val_idx])]

                for tok in tmpl:
                    if tok == pop.x_id:
                        seqs[i, pos : pos + len(name_tokens)] = name_tokens
                        masks[i, pos : pos + len(name_tokens)] = 7
                        pos += len(name_tokens)
                    elif tok == pop.y_id:
                        seqs[i, pos : pos + len(val)] = val
                        masks[i, pos] = k + 1
                        masks[i, pos + 1 : pos + len(val)] = -(k + 1)
                        pos += len(val)
                    else:
                        seqs[i, pos] = tok
                        masks[i, pos] = 7
                        pos += 1
                if self.lang_switch > 0 and rng.random() < self.lang_switch:
                    cur_fr = 1 - cur_fr

            masks[i, pos:] = 8  # end-of-document region: train PAD emission

        return seqs, masks, persons, langs

    def get_batch(self, rng: jax.Array, batch_size: int) -> dict:
        """Training batch from cache (uses internal numpy RNG, ignores rng)."""
        if self._cache is None or self._cache_idx + batch_size > self._cache[0].shape[0]:
            self._cache = self._generate(batch_size * self.CACHE_MULTIPLIER, eval_templates=False, rng=self._np_rng)
            self._cache_idx = 0
        seqs, masks, persons, _ = self._cache
        sl = slice(self._cache_idx, self._cache_idx + batch_size)
        self._cache_idx += batch_size
        self.exposure_counts += np.bincount(persons[sl], minlength=self.pop.num_people)
        return {
            "inputs": jnp.array(seqs[sl, :-1]),
            "targets": jnp.array(seqs[sl, 1:]),
            "mask": jnp.array(masks[sl, 1:]),
        }

    def get_prompts(self, batch_size: int) -> tuple[jax.Array, np.ndarray]:
        """Generation prompts [BOS, name..., PAD...] for randomly sampled people."""
        pop = self.pop
        persons = self.person_ids[self._np_rng.integers(len(self.person_ids), size=batch_size)]
        prompts = np.full((batch_size, self.prompt_len), pop.pad_id, dtype=np.int32)
        prompts[:, 0] = pop.bos_id
        for i, p in enumerate(persons):
            name_tokens = pop.name_tokens(p)
            prompts[i, 1 : 1 + len(name_tokens)] = name_tokens
        return jnp.array(prompts), persons

    @staticmethod
    def loss_fn(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
        """Cross-entropy over real tokens and the end-of-document PAD region."""
        return compute_cross_entropy(logits, targets, (mask != 0).astype(jnp.float32))

    def evaluate(self, forward_fn, batch_size: int = 256) -> dict[str, float]:
        """Exact metrics over the full eval set, streamed in batches."""
        n = self.eval_inputs.shape[0]
        # Per-language split only when biographies are monolingual (with
        # sentence switching the per-biography start-language label is wrong)
        per_lang = 0.0 < self.lang_mix < 1.0 and self.lang_switch == 0.0
        total = None
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            logits = forward_fn(jnp.array(self.eval_inputs[sl]))
            targets, mask = jnp.array(self.eval_targets[sl]), jnp.array(self.eval_mask[sl])
            sums = self._metric_sums(logits, targets, mask)
            if per_lang:
                fr = jnp.array(self._eval_langs[sl])[:, None]
                for lang, rows in [("fr", fr), ("en", 1 - fr)]:
                    lang_sums = self._metric_sums(logits, targets, mask * rows)
                    sums.update({f"{lang}/{k}": v for k, v in lang_sums.items() if k.startswith("first_token_")})
            total = sums if total is None else {k: total[k] + v for k, v in sums.items()}
        return self._finalize_metrics(total)

    # Token categories for metrics, aggregated over all attribute types.
    # Role 8 (trained end-of-document padding) is excluded everywhere so the
    # metrics keep measuring real-token prediction only.
    METRIC_CATEGORIES = {
        "": lambda mask: (mask != 0) & (mask != 8),
        "attribute_": lambda mask: (mask != 0) & (mask != 7) & (mask != 8),
        "first_token_": lambda mask: (mask >= 1) & (mask <= NUM_ATTRIBUTES),
    }

    def _metric_sums(self, logits, targets, mask) -> dict[str, float]:
        """Unnormalized loss/correct/count sums per token category."""
        per_token = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        correct = (jnp.argmax(logits, axis=-1) == targets).astype(jnp.float32)
        sums = {}
        for key, category in self.METRIC_CATEGORIES.items():
            m = category(mask).astype(jnp.float32)
            sums[f"{key}loss"] = float(jnp.sum(per_token * m))
            sums[f"{key}correct"] = float(jnp.sum(correct * m))
            sums[f"{key}count"] = float(jnp.sum(m))
        return sums

    def _finalize_metrics(self, sums: dict[str, float]) -> dict[str, float]:
        metrics = {}
        for key in self.METRIC_CATEGORIES:
            count = max(sums[f"{key}count"], 1.0)
            metrics[f"{self.name}/{key}loss"] = sums[f"{key}loss"] / count
            metrics[f"{self.name}/{key}accuracy"] = sums[f"{key}correct"] / count
        # Total knowledge held for this dataset: exact-recall rate (first value
        # token) times the full population size. Eval accuracy on a capped
        # sample is unbiased, so acc * |people| estimates facts-recalled at scale.
        metrics[f"{self.name}/knowledge"] = (
            metrics[f"{self.name}/first_token_accuracy"] * len(self.person_ids)
        )
        # Per-template-language exact recall (only when the eval set mixes both)
        for lang in ("fr", "en"):
            if f"{lang}/first_token_count" in sums:
                count = max(sums[f"{lang}/first_token_count"], 1.0)
                metrics[f"{self.name}_{lang}/first_token_loss"] = sums[f"{lang}/first_token_loss"] / count
                metrics[f"{self.name}_{lang}/first_token_accuracy"] = sums[f"{lang}/first_token_correct"] / count
        return metrics
