"""Preprocess biography data: GPT-2 BPE tokenization + vocab pruning to used tokens.

Tokenizes templates, attribute values, and names with the pretrained GPT-2
tokenizer (plus special and year tokens), then remaps the vocabulary to only
the tokens actually used so the model embedding stays small.

Run: uv run python -m src.data.preprocess_biography
"""
import argparse
import os

import numpy as np
from tokenizers import Tokenizer

from src.data.biography_corpus import (
    ATTRIBUTE_NAMES, TEMPLATES, TEMPLATES_FR, CITIES, UNIVERSITIES, FIELDS,
    COMPANIES, BIRTH_YEARS, FIRST_NAMES, MIDDLE_NAMES, LAST_NAMES,
)

SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[X]", "[Y]"]


def load_tokenizer():
    """Load pretrained GPT-2 tokenizer and add special + year tokens."""
    tokenizer = Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.add_tokens(BIRTH_YEARS)  # force single-token years
    return tokenizer


def pad_to_matrix(all_ids, pad_id):
    """Pad a ragged list of token-id lists into a matrix, return (padded, lengths)."""
    lengths = np.array([len(ids) for ids in all_ids], dtype=np.int32)
    padded = np.full((len(all_ids), int(lengths.max())), pad_id, dtype=np.int32)
    for i, ids in enumerate(all_ids):
        padded[i, : len(ids)] = ids
    return padded, lengths


def tokenize_value_list(tokenizer, values, pad_id):
    """Tokenize a list of strings, return (padded ids, lengths)."""
    return pad_to_matrix([tokenizer.encode(v).ids for v in values], pad_id)


def filter_unique_first_tokens(tokenizer, values):
    """Keep only values whose first BPE token is unique within the pool."""
    seen, filtered = set(), []
    for v in values:
        ids = tokenizer.encode(v).ids
        if ids and ids[0] not in seen:
            seen.add(ids[0])
            filtered.append(v)
    return filtered


def tokenize_templates(tokenizer, templates, x_id, y_id, pad_id):
    """Tokenize templates keeping [X]/[Y] as single placeholder tokens."""
    all_ids = []
    for template in templates:
        parts, remaining = [], template
        while remaining:
            positions = [(remaining.find(p), tid) for p, tid in [("[X]", x_id), ("[Y]", y_id)]]
            positions = [(pos, tid) for pos, tid in positions if pos != -1]
            if not positions:
                if remaining.strip():
                    parts.extend(tokenizer.encode(remaining).ids)
                break
            pos, tid = min(positions)
            if remaining[:pos].strip():
                parts.extend(tokenizer.encode(remaining[:pos]).ids)
            parts.append(tid)
            remaining = remaining[pos + 3:]
        all_ids.append(parts)

    return pad_to_matrix(all_ids, pad_id)


def main():
    parser = argparse.ArgumentParser(description="Preprocess biography data")
    parser.add_argument("--output_dir", type=str, default="data/biography")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = load_tokenizer()
    pad_id, bos_id, x_id, y_id = (tokenizer.token_to_id(t) for t in SPECIAL_TOKENS)

    for year in ["1800", "1900", "2020"]:
        assert len(tokenizer.encode(year).ids) == 1, f"Year '{year}' is not a single token"

    value_pools = {
        "birthdate": BIRTH_YEARS,
        "birthplace": CITIES,
        "university": UNIVERSITIES,
        "major": FIELDS,
        "company": COMPANIES,
        "work_location": CITIES,
    }

    arrays = {}
    num_values_per_attr = []
    for k, attr_name in enumerate(ATTRIBUTE_NAMES):
        filtered = filter_unique_first_tokens(tokenizer, value_pools[attr_name])
        num_values_per_attr.append(len(filtered))
        print(f"{attr_name}: {len(value_pools[attr_name])} -> {len(filtered)} values after first-token filter")

        value_ids, value_lengths = tokenize_value_list(tokenizer, filtered, pad_id)
        arrays[f"attr_{k}_value_strings"] = np.array(filtered, dtype=object)
        arrays[f"attr_{k}_value_token_ids"] = value_ids
        arrays[f"attr_{k}_value_token_lengths"] = value_lengths
        arrays[f"attr_{k}_first_token_ids"] = value_ids[:, 0].copy()

        assert len(TEMPLATES_FR[attr_name]) == len(TEMPLATES[attr_name])
        for suffix, templates in [("", TEMPLATES), ("_fr", TEMPLATES_FR)]:
            tmpl_ids, tmpl_lengths = tokenize_templates(tokenizer, templates[attr_name], x_id, y_id, pad_id)
            arrays[f"attr_{k}_template{suffix}_token_ids"] = tmpl_ids
            arrays[f"attr_{k}_template{suffix}_token_lengths"] = tmpl_lengths

    arrays["num_values_per_attr"] = np.array(num_values_per_attr, dtype=np.int32)

    for pool_name, pool in [("first_name", FIRST_NAMES), ("middle_name", MIDDLE_NAMES), ("last_name", LAST_NAMES)]:
        ids, lengths = tokenize_value_list(tokenizer, pool, pad_id)
        arrays[f"{pool_name}_token_ids"] = ids
        arrays[f"{pool_name}_token_lengths"] = lengths

    # --- Vocab pruning: remap to only used token ids ---
    # English tokens keep their historical positions; French-only template
    # tokens are appended after, so English-only token ids stay stable.
    specials = {pad_id, bos_id, x_id, y_id}
    used_en, used_fr = set(), set()
    for key, arr in arrays.items():
        if key.endswith("_token_ids"):
            (used_fr if "_fr_" in key else used_en).update(np.unique(arr).tolist())
    kept = ([pad_id, bos_id, x_id, y_id] + sorted(used_en - specials)
            + sorted(used_fr - used_en - specials))
    old_to_new = np.full(max(kept) + 1, -1, dtype=np.int32)
    old_to_new[kept] = np.arange(len(kept), dtype=np.int32)

    for key in list(arrays):
        if key.endswith("_token_ids"):
            arrays[key] = old_to_new[arrays[key]]
    arrays["token_strings"] = np.array([tokenizer.id_to_token(i) for i in kept], dtype=object)
    arrays["vocab_size"] = len(kept)
    arrays["english_vocab_size"] = 4 + len(used_en - specials)  # English-only prefix
    arrays["pad_id"], arrays["bos_id"], arrays["x_id"], arrays["y_id"] = 0, 1, 2, 3
    print(f"\nPruned vocab: {tokenizer.get_vocab_size()} -> {len(kept)} tokens")

    # --- Length metadata ---
    max_name_len = sum(
        int(arrays[f"{p}_token_lengths"].max()) for p in ["first_name", "middle_name", "last_name"]
    )
    for suffix in ["", "_fr"]:
        max_sentence_len = 0
        for k in range(6):
            tmpl_ids = arrays[f"attr_{k}_template{suffix}_token_ids"]
            tmpl_lengths = arrays[f"attr_{k}_template{suffix}_token_lengths"]
            max_val_len = int(arrays[f"attr_{k}_value_token_lengths"].max())
            for t in range(tmpl_ids.shape[0]):
                row = tmpl_ids[t, : int(tmpl_lengths[t])]
                n_placeholders = int(np.sum((row == 2) | (row == 3)))
                n_names = int(np.sum(row == 2))
                sentence_len = len(row) - n_placeholders + n_names * max_name_len + max_val_len
                max_sentence_len = max(max_sentence_len, sentence_len)
        arrays[f"max_sentence_len{suffix}"] = max_sentence_len
        print(f"max_sentence_len{suffix}: {max_sentence_len}")

    arrays["max_name_len"] = max_name_len
    print(f"max_name_len: {max_name_len}")

    output_path = os.path.join(args.output_dir, "preprocessed.npz")
    np.savez(output_path, **arrays)
    print(f"Saved preprocessed data to {output_path}")


if __name__ == "__main__":
    main()
