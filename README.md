# forgetting-dynamics

Code for the experiments of the paper. Run all commands from the repository root unless stated otherwise.

Requirements: Python 3.10+ with `jax`, `flax`, `optax`, `numpy`, `matplotlib`, `tokenizers` and `tqdm` for the minimal model and the transformer; `torch` and `transformers` for the pretrained language model.

```
mkdir -p plots
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
python -m llm.gate_a_set --out llm/out/gate.json
python -m llm.build_b --n_people 250 --out_dir llm/out
python -m llm.inject --lr 1e-5 --steps 400
```
