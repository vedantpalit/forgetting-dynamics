# forgetting-dynamics

Code for the experiments of the paper. Run all commands from the repository root unless stated otherwise; the JAX commands below assume the environment is active (or prefix them with `uv run`), and the language-model commands use `.venv-llm/bin/python`.

## Installation

The minimal model and the transformer use JAX; install them and the plotting tools with [uv](https://docs.astral.sh/uv/):

```
uv sync --group analysis
mkdir -p plots
```

The pretrained language model uses PyTorch, in a separate environment:

```
uv venv .venv-llm
uv pip install --python .venv-llm --group llm
```

## Minimal model

Collapse, recovery and erosion of the old facts in the associative memory:

```
cd toy_final_iclr
python paper_base_runs.py --seeds 0-9 --out paper_base.json
python plot_paper_base.py
```

## Transformer

Pretrain on the two sets of old facts, then finetune on new facts whose answers lie in one region (`disjoint`), or in both regions for comparison (`all_values`):

```
python -m src.experiments.mlpfree_pretrain
python -m src.experiments.mlpfree_injection --pretrain_step 16000 --inject_seed 0 --inject_condition disjoint
python -m src.experiments.mlpfree_injection --pretrain_step 16000 --inject_seed 0 --inject_condition all_values
```

## Pretrained language model

Select the old facts OLMo 2 1B knows, build the new facts, and finetune:

```
.venv-llm/bin/python -m llm.gate_a_set --out llm/out/gate.json
.venv-llm/bin/python -m llm.build_b --n_people 250 --out_dir llm/out
.venv-llm/bin/python -m llm.inject --lr 1e-5 --steps 400
```

## Additional experiments

Replay in the transformer. A fixed subset of 250 old individuals is replayed and recall is read on the others (`dataA` in the log); `--arm` sets the schedule (`none`, `every`, `int25`, `win60_120`, `late1000`) and `--source C` replays new individuals whose answers lie in the region of the old facts:

```
python -m src.experiments.mlpfree_replay_injection --arm win60_120 --budget 9600 --total 1200 --inject_seed 0 --lr_mult 1 --inject_batch 256
```

Comparisons for the weight edit in OLMo 2 1B (interpolation toward the pretrained weights, shrinkage to the same norm, other directions of the same size, a random direction), with complete-answer exact match and the shared part of the edit. The small set of synthetic individuals is built with `--tag small`, the real entities with `llm.build_b_entityq --relation P176`, and the generic text with `python -m llm_real.run fetch-generic`:

```
.venv-llm/bin/python -m llm.build_b --n_people 250 --tag small --out_dir llm/out
.venv-llm/bin/python -m llm.edit_comparisons --lr 1e-5 --steps 400 --seed 0 --facts llm/out/b_facts_small.json --out llm/out/edit_small_s0.json
.venv-llm/bin/python -m llm.edit_comparisons --lr 1e-5 --steps 400 --seed 0 --facts llm/out/b_facts_eq_P176.json --out llm/out/edit_real_s0.json
```

Subjects of the new facts. Manufacturers permuted among the real products, and invented names with the same answers, with or without a surname shared by the products of one manufacturer:

```
.venv-llm/bin/python -m llm.permute_answers --seed 0
.venv-llm/bin/python -m llm.invented_subjects --seed 0
.venv-llm/bin/python -m llm.edit_comparisons --lite --lr 1e-5 --steps 800 --eval_steps 0,80,150,200,300,400,600,800 --seed 0 --facts llm/out/b_facts_eq_P176_perm_s0.json --out llm/out/perm_s0.json
```

First tokens shared between answers:

```
.venv-llm/bin/python -m llm.first_token_overlap
```

A larger model, OLMo 2 7B, through the same steps:

```
.venv-llm/bin/python -m llm.gate_a_set --model_id allenai/OLMo-2-1124-7B --out llm/out/gate_7b.json
.venv-llm/bin/python -m llm.build_b --n_people 250 --tag small_7b --gate llm/out/gate_7b.json --model_id allenai/OLMo-2-1124-7B --out_dir llm/out
.venv-llm/bin/python -m llm.inject --lr 1e-5 --steps 400 --model_id allenai/OLMo-2-1124-7B --gate llm/out/gate_7b.json --facts llm/out/b_facts_small_7b.json
```

Replay in OLMo 2 1B. Sentences stating one eighth of the old facts are added to the finetuning data, at every step (`spread`) or in steps 80 to 139 (`burst`), and recall is read on the never-replayed facts:

```
.venv-llm/bin/python -m llm.inject_replay --lr 1e-5 --steps 400 --seed 0 --cond none --replay_seqs 0
.venv-llm/bin/python -m llm.inject_replay --lr 1e-5 --steps 400 --seed 0 --cond spread --replay_seqs 2 --replay_from 1 --replay_to 400
.venv-llm/bin/python -m llm.inject_replay --lr 1e-5 --steps 400 --seed 0 --cond burst --replay_seqs 13 --replay_from 80 --replay_to 139
```
