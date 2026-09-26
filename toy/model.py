"""The linear store (PLAN.md "Model, stage 1"): W in R^{d x |V|}, b in R^{|V|},
zero-initialized, logits = keys @ W + b. Stage 1 only (frozen embeddings,
keys precomputed by populations.py) — stage 2 (trainable embeddings) is
explicitly the lowest-priority ablation in PLAN.md §9 ("only if stage 1's
verdict and time budget justify it") and is not built here; it would need
populations.py to additionally expose (first_idx, last_idx) index arrays
and the raw embedding tables so gradients can flow into them, which this
module deliberately does not add yet.
"""

import numpy as np


class LinearStore:
    def __init__(self, d, v_total):
        self.W = np.zeros((d, v_total))
        self.b = np.zeros(v_total)

    def logits(self, keys):
        return keys @ self.W + self.b

    def copy(self):
        out = LinearStore(self.W.shape[0], self.W.shape[1])
        out.W = self.W.copy()
        out.b = self.b.copy()
        return out
