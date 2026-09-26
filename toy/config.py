"""Config schema for the toy (PLAN.md §4). Every knob is explicit; nothing
defaults silently at the population/model/train layer — those layers only
ever read fields off these dataclasses. All cross-field invariants are
asserted in __post_init__, not left to be eyeballed (PLAN.md task
requirement 4).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class EmbeddingConfig:
    # d=2048 with A u D u B = 1440 keys (70% of d) keeps every key linearly
    # independent with real margin, so an interference-free storage solution
    # provably exists and no measured erosion can be capacity (PLAN.md §0.3a).
    d: int = 2048
    # 48 (A) + 24 (D) + 24 (B non-overlap) = 96 reserved tokens, of 128.
    n_first: int = 128
    n_last: int = 128
    # Compositionality (PLAN.md §0.1). k = alpha*(e_f+e_l)/sqrt(2) + sqrt(1-alpha^2)*u.
    # alpha=1 is the purely additive model, whose key span is |F|+|L|-1 REGARDLESS of d
    # and which the ridge pre-check shows is structurally infeasible (rank 148, ridge
    # accuracy 0.74). alpha=0 is private keys with no erosion channel. Shared-last-name
    # key cosine is ~alpha^2/2, the quantity the Gram check measures.
    alpha: float = 0.7
    # Shared MEAN direction (PLAN.md §0.11). k_i = sqrt(beta)*mu + sqrt(1-beta)*(above),
    # with mu one frozen unit vector common to EVERY key. beta=0 is exactly the previous
    # construction. This is not a second copy of `feature_shared_strength`: rho correlates
    # the name-part FEATURES with each other inside a low-rank subspace, leaving the mean
    # key near zero; beta gives the key set a large common mean, so k_bar_B has order-one
    # norm and a W update along it produces an individual-INDEPENDENT logit shift. That is
    # the quantity the transformer measurement found carries suppression, and which the
    # toy at beta=0 cannot produce at all (its induced constant shift was 0.00059).
    beta: float = 0.0
    # Feature geometry knob, PLAN.md §4.3.
    feature_shared_strength: float = 0.0  # rho in [0, 1]; 0 = idealized i.i.d.
    rank_shared: int = 8

    def __post_init__(self):
        assert self.d > 0
        assert self.n_first > 0 and self.n_last > 0
        assert 0.0 <= self.alpha <= 1.0
        assert 0.0 <= self.beta < 1.0, "beta=1 makes every key identical"
        assert 0.0 <= self.feature_shared_strength <= 1.0
        assert 0 < self.rank_shared <= self.d


@dataclass(frozen=True)
class VocabConfig:
    v_a_size: int = 256
    v_b_size: int = 256

    @property
    def v_total(self):
        return self.v_a_size + self.v_b_size


@dataclass(frozen=True)
class PopulationConfig:
    # Sized so every overlap grid point lands on an integer token count AND an
    # integer B-overlapping count: 48 A tokens x {0,.25,.5,.75,1} = 0/12/24/36/48
    # and 240 x the same = 0/60/120/180/240 (PLAN.md §0.3b).
    n_a: int = 960
    n_ballast: int = 240
    n_b: int = 240
    n_c: int = 240
    ballast_present: bool = True
    # Concentration: number of distinct V_B values assigned to B (PLAN.md §4.5).
    concentration: int = 8
    # Overlap: fraction of B whose last-name token is drawn from A's last names (§4.6).
    overlap: float = 0.5
    # Name-token multiplicity (PLAN.md §4.4a addition): how many individuals
    # within a population share a given first- or last-name token, held
    # exactly fixed by construction (round-robin assignment, not sampled) —
    # not incidental: the main biography experiments found damage attaches
    # to *shared tokens*, not individuals, so how many individuals a single
    # touched token drags down is a real mechanism variable, and leaving it
    # to random pool coverage would make it vary across conditions for no
    # reason. Applies to A's first- and last-name pools.
    name_multiplicity: int = 20
    # D and B use their own, smaller multiplicity. Their internal token sharing is
    # NOT a mechanism variable — only A's is, since only A is measured — and at 20
    # their pools would be 12, below the round-robin construction's pool >= multiplicity
    # requirement. 10 gives pools of 24 each.
    aux_multiplicity: int = 10

    def __post_init__(self):
        assert self.n_a > 0 and self.n_b > 0 and self.n_c > 0
        assert self.n_ballast >= 0
        assert self.concentration >= 1
        assert 0.0 <= self.overlap <= 1.0
        assert self.name_multiplicity > 0
        assert self.n_a % self.name_multiplicity == 0, \
            "n_a must divide evenly by name_multiplicity for exact round-robin multiplicity"
        if self.ballast_present:
            assert self.n_ballast % self.name_multiplicity == 0
        assert self.n_b % self.name_multiplicity == 0
        # The round-robin pairing trick (populations._round_robin_pairs) needs
        # pool_size >= multiplicity, or the modular block-shift wraps around
        # and produces duplicate pairs — this is exactly the failure mode a
        # too-small reduced-scale smoke config hit during implementation.
        m, a = self.name_multiplicity, self.aux_multiplicity
        assert self.n_a // m >= m, f"n_a={self.n_a} too small for multiplicity={m} (pool would be smaller than multiplicity)"
        if self.ballast_present:
            assert self.n_ballast // a >= a, f"n_ballast={self.n_ballast} too small for aux_multiplicity={a}"
        assert self.n_b // a >= a, f"n_b={self.n_b} too small for aux_multiplicity={a}"

    @property
    def a_pool_size(self):
        return self.n_a // self.name_multiplicity

    @property
    def d_pool_size(self):
        return self.n_ballast // self.aux_multiplicity if self.ballast_present else 0

    @property
    def b_own_pool_size(self):
        """Size of B's reserved (non-overlap-eligible, always-first-name) sub-pool."""
        return self.n_b // self.aux_multiplicity

    @property
    def n_touched_tokens(self):
        """How many of A's last-name tokens B actually touches (PLAN.md §0.3b).

        Overlap now controls TOKEN COVERAGE directly rather than the fraction of B
        drawing from A's pool. The two decouple: with B drawing uniformly, even a
        quarter of B covered most of A's tokens by the coupon-collector effect, which
        left 6 untouched A individuals at overlap 1.0 and destroyed the touched/untouched
        split's power at the top of the grid.
        """
        return int(round(self.overlap * self.a_pool_size))

    @property
    def n_overlapping_b(self):
        return int(round(self.overlap * self.n_b))

    @property
    def collisions_per_touched_token(self):
        """Injection pressure per touched token. CONSTANT across the overlap grid by
        construction, since both the numerator and denominator scale with overlap --
        which is what keeps coverage from being confounded with per-token intensity."""
        t = self.n_touched_tokens
        return (self.n_overlapping_b / t) if t else 0.0

    def required_first_tokens(self):
        return self.a_pool_size + self.d_pool_size + self.b_own_pool_size

    def required_last_tokens(self):
        return self.a_pool_size + self.d_pool_size + self.b_own_pool_size


@dataclass(frozen=True)
class OptimizerConfig:
    optimizer: str = "sgd"  # {"sgd", "adam", "sgd_minibatch"}
    lr: float = 3.0  # pretraining LR, from the gate (PLAN.md §0.4a)
    # Injection runs at lr / inject_lr_ratio, mirroring the real experiments, where the
    # reference protocol pairs a 4e-4 pretraining peak with a constant 3e-5 during
    # injection: 4e-4 / 3e-5 = 40/3. Hardcoded rather than imported -- toy/ is standalone
    # and does not depend on the biography pipeline -- but the VALUE is theirs, chosen
    # externally and independently of anything this toy will show.
    #
    # This is load-bearing, not fidelity polish (PLAN.md §0.4b). s_eff scales with
    # injection LR, and Addition 3 established that restricted rank is untouched below
    # the critical push s* and destroyed above it, so injection LR decides which side of
    # that transition the battery runs on. At the pretraining LR the update would very
    # likely be far supercritical, and CHECK 1 would fail because nothing survives rather
    # than because suppression lacks a rank-sparing character. It also protects CHECK 3
    # and CHECK 4: if injection floors accuracy at zero for every concentration and
    # overlap, trough depth saturates and a floor is indistinguishable from an invariance.
    inject_lr_ratio: float = 40.0 / 3.0
    minibatch_size: Optional[int] = None  # required iff optimizer == "sgd_minibatch"

    def __post_init__(self):
        assert self.optimizer in ("sgd", "adam", "sgd_minibatch")
        assert self.lr > 0
        assert self.inject_lr_ratio > 0
        if self.optimizer == "sgd_minibatch":
            assert self.minibatch_size is not None and self.minibatch_size > 0
        else:
            assert self.minibatch_size is None

    @property
    def inject_lr(self):
        """The rate injection actually runs at. Pretraining uses `lr`."""
        return self.lr / self.inject_lr_ratio


@dataclass(frozen=True)
class ProtocolConfig:
    stage: int = 1  # {1: frozen embeddings, 2: trainable}
    pretrain_ceiling_acc: float = 0.99
    pretrain_ceiling_window: int = 20
    pretrain_extra_fraction: float = 0.25
    pretrain_max_steps: int = 20000
    finetune_flat_window: int = 50
    finetune_flat_rank_range: float = 0.02
    finetune_flat_acc_range: float = 0.01
    finetune_min_b_acc: float = 0.99
    finetune_max_steps: int = 5000
    eval_every_dense: int = 1  # every step for the first `eval_dense_until` steps
    eval_dense_until: int = 200
    eval_every_sparse: int = 10  # every N steps after that

    def __post_init__(self):
        assert self.stage in (1, 2)
        assert 0 < self.pretrain_ceiling_acc <= 1.0
        assert self.pretrain_ceiling_window > 0
        assert self.pretrain_extra_fraction >= 0
        assert self.pretrain_max_steps > 0
        assert self.finetune_flat_window > 0
        assert self.finetune_max_steps > 0


@dataclass(frozen=True)
class ToyConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vocab: VocabConfig = field(default_factory=VocabConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    seed: int = 0
    loss: str = "ce"  # {"ce", "mse"} — PLAN.md ablation 1

    def __post_init__(self):
        assert self.loss in ("ce", "mse")
        req_first = self.population.required_first_tokens()
        req_last = self.population.required_last_tokens()
        assert self.embedding.n_first >= req_first, (
            f"n_first={self.embedding.n_first} too small for A/D/B's reserved "
            f"sub-pools (need >= {req_first} at name_multiplicity="
            f"{self.population.name_multiplicity})"
        )
        assert self.embedding.n_last >= req_last, (
            f"n_last={self.embedding.n_last} too small for A/D/B's reserved "
            f"sub-pools (need >= {req_last} at name_multiplicity="
            f"{self.population.name_multiplicity})"
        )

    def split_seeds(self):
        """One seed controls population construction and training jointly
        (PLAN.md §10.3): split deterministically so both halves are
        reproducible from the single top-level seed."""
        return self.seed, self.seed + 1_000_000

    def condition_id(self):
        e, p, o = self.embedding, self.population, self.optimizer
        return (
            f"overlap{p.overlap:.2f}_conc{p.concentration}_ballast{int(p.ballast_present)}_"
            f"rho{e.feature_shared_strength:.2f}_{o.optimizer}_lr{o.lr:g}_stage{self.protocol.stage}"
        )
