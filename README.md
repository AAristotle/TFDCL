# TFDCL: Time-aware Fact Diffusion with Contrastive Learning for Temporal Knowledge Graph Reasoning

Official implementation of the ECAI 2025 paper.

## Overview

**TFDCL** is a novel framework for **Temporal Knowledge Graph (TKG) Reasoning** (extrapolation setting). Given a query $(s, r, ?, t_q)$, the model predicts the missing entity at a future timestamp by leveraging rich historical information.

The framework consists of three core modules:

1. **Temporal Feature Encoding** — Captures short-term and long-term entity/relation evolution from historical snapshots using R-GCN with a relation-guided filtering mechanism and GRU-based temporal aggregation.
2. **Time-aware Fact Diffusion (TFD)** — Injects Gaussian noise into fact-level representations and progressively denoises them via a Transformer encoder, improving generalization to unseen events.
3. **Short-term and Long-term Contrastive Learning** — Aligns short-term and long-term query representations using supervised contrastive loss, enhancing robustness and representation quality.

---

## Requirements

```bash
pip install torch dgl transformers numpy pandas tqdm scipy
```

Key dependencies:
- Python >= 3.8
- PyTorch >= 1.10
- DGL >= 0.8
- Transformers >= 4.0
- CUDA (recommended: NVIDIA A100 40GB)

---

## Datasets

We evaluate on four benchmark TKG datasets:

| Dataset     | #Entities | #Relations | #Train    | #Valid  | #Test   | #Snapshots |
|-------------|-----------|------------|-----------|---------|---------|------------|
| ICEWS14     | 6,869     | 230        | 74,845    | 8,514   | 7,371   | 365        |
| ICEWS18     | 10,094    | 256        | 373,018   | 45,995  | 49,545  | 365        |
| ICEWS05-15  | 23,033    | 251        | 368,868   | 46,302  | 46,159  | 4,017      |
| GDELT       | 7,691     | 240        | 1,734,399 | 238,765 | 305,241 | 2,975      |

Raw data files (`train.txt`, `valid.txt`, `test.txt`, `entity2id.txt`, `relation2id.txt`) should be placed under `data/<DATASET>/`.

### Preprocessing

Before training, generate the required historical subgraph and sequence files:

```bash
# Generate historical subgraphs
python data/get_his_subg.py --dataset ICEWS14

# Generate entity-word graph (for static graph)
python data/ICEWS14/ent2word.py
```

---

## Pre-trained Language Model Embeddings

TFDCL uses a pre-trained language model (BERT, T5, etc.) to encode entity and relation text descriptions. Run the following to generate and cache embeddings before training:

```bash
python main.py -d ICEWS14 \
    --model-type bert \
    --plm bert-large-cased \
    --num-k 5
```

Embeddings will be saved to `data/<DATASET>/text_emb/<plm>/`.

---

## Training

### Example: ICEWS14

```bash
python main.py \
    -d ICEWS14 \
    --model-type bert \
    --plm bert-large-cased \
    --gpu 0 \
    --encoder uvrgcn \
    --decoder convtranse \
    --n-hidden 200 \
    --lr 0.001 \
    --n-epochs 30 \
    --train-history-len 7 \
    --add-static-graph \
    --use-cl \
    --use-cd \
    --temperature 0.03 \
    --pre-weight 0.7 \
    --pre-type all \
    --layer-norm \
    --diffusion_steps 10 \
    --num-k 5
```

### Example: ICEWS18

```bash
python main.py \
    -d ICEWS18 \
    --model-type bert \
    --plm bert-large-cased \
    --gpu 0 \
    --encoder uvrgcn \
    --decoder convtranse \
    --lr 0.001 \
    --n-epochs 15 \
    --n-layers 2 \
    --train-history-len 10 \
    --test-history-len 10 \
    --max_len 128 \
    --diffusion_max_len 128 \
    --timestamps 304 \
    --dilate-len 1 \
    --evaluate-every 1 \
    --self-loop \
    --layer-norm \
    --add-static-graph \
    --use-cl \
    --use-cd \
    --entity-prediction \
    --weight 0.5 \
    --pre-weight 0.7 \
    --pre-type all \
    --temperature 0.03 \
    --angle 10 \
    --discount 1 \
    --num-k 5
```

### Key Arguments

| Argument | Description | Recommended |
|---|---|---|
| `-d` | Dataset (`ICEWS14`, `ICEWS18`, `ICEWS05-15`, `GDELT`) | — |
| `--model-type` | Pre-trained LM type (`bert`, `t5`, `gpt2`) | `bert` *(required)* |
| `--plm` | Pre-trained LM name or path | `bert-large-cased` *(required)* |
| `--encoder` | GNN encoder (`uvrgcn`, `kbat`, `compgcn`) | `uvrgcn` |
| `--decoder` | Decoder type | `convtranse` |
| `--n-hidden` | Hidden dimension size | `200` |
| `--n-layers` | Number of RGCN layers | `2` |
| `--lr` | Learning rate | `0.001` |
| `--train-history-len` | Length of short-term historical snapshots | `7` (ICEWS14) / `10` (others) |
| `--add-static-graph` | Use static entity-word graph | enabled |
| `--use-cl` | Enable contrastive learning | enabled |
| `--use-cd` | Enable cross-dimension module | enabled |
| `--layer-norm` | Apply layer normalization in RGCN | enabled |
| `--temperature` | Contrastive learning temperature τ | `0.03` |
| `--pre-weight` | Weight α for long/short-term fusion | `0.7` |
| `--pre-type` | History fusion mode (`long`, `short`, `all`) | `all` |
| `--weight` | Static constraint weight | `0.5` / `1.0` |
| `--angle` | Evolution speed | `10` |
| `--discount` | Static constraint discount factor | `1` |
| `--diffusion_steps` | Number of diffusion steps M | `10` |
| `--max_len` / `--diffusion_max_len` | History sequence length for diffusion | `128` |
| `--num-k` | Number of neighbors for DSEP | `5` |

---

## Testing

```bash
python main.py \
    -d ICEWS14 \
    --model-type bert \
    --plm bert-large-cased \
    --gpu 0 \
    --test \
    --add-static-graph \
    --use-cl \
    --use-cd \
    --layer-norm \
    --temperature 0.03 \
    --pre-weight 0.7
```

> **Note:** The testing command must use the same model-architecture flags (`--use-cd`, `--layer-norm`, `--add-static-graph`, `--use-cl`, etc.) as the training command; otherwise, the saved checkpoint may fail to load.

Results are saved to `result/<DATASET>.csv`. Case study outputs are saved to `result/tfdcl_case_study_<DATASET>.txt`.

---

## Main Results

Performance comparison (MRR and Hits@N, %) on four benchmark datasets:

| Model | ICEWS14 MRR | ICEWS18 MRR | ICEWS05-15 MRR | GDELT MRR |
|---|---|---|---|---|
| RE-GCN (2021) | 40.39 | 30.58 | 48.03 | 19.64 |
| TiRGN (2022) | 44.04 | 33.66 | 50.04 | 21.67 |
| HisMatch (2022) | 46.42 | 33.99 | 52.85 | 22.01 |
| LogCL (2023) | 48.87 | 35.67 | 57.04 | 23.75 |
| Re-Temp (2023) | 48.04 | **35.82** | 56.30 | **25.05** |
| TKGR-CPRSCL (2025) | _51.13_ | - | _58.65_ | 24.22 |
| **TFDCL (Ours)** | **52.34** | **39.21** | **60.57** | **26.37** |

Bold = best, *italic* = second best. TFDCL achieves state-of-the-art results on all four datasets.

---

## Project Structure

```
TFDCL/
├── main.py                  # Training & evaluation entry point
├── DSE.py                   # Dynamic Semantic Encoding (PLM embeddings)
├── difffu_21.py             # Time-aware Fact Diffusion module
├── knowledge_graph.py       # Knowledge graph utilities
├── utils_entity.py          # Entity utility functions
├── unseen_event.py          # Unseen event analysis
├── tri2seq.py               # Triplet to sequence conversion
├── src/
│   ├── model.py             # Base R-GCN model
│   ├── rrgcn.py             # Recurrent R-GCN (RecurrentRGCN)
│   ├── decoder.py           # Decoder modules
│   └── decoder2.py          # ConvTransE / ConvTransR decoders
├── rgcn/
│   ├── layers.py            # RGCN / RGAT / CompGCN layers
│   ├── utils.py             # Graph building & evaluation utilities
│   ├── knowledge_graph.py   # KG data loader
│   ├── vocabulary.py        # Vocabulary module
│   └── dsf.py               # DSF embedding module
├── data/
│   ├── ICEWS14/
│   ├── ICEWS18/
│   ├── ICEWS05-15/
│   └── get_his_subg.py      # Historical subgraph preprocessing
└── result/                  # Output CSV and case study files
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{geng2025tfdcl,
  title     = {Time-aware Fact Diffusion with Contrastive Learning for Temporal Knowledge Graph Reasoning},
  author    = {Geng, Rushan and Luo, Cuicui},
  booktitle = {Proceedings of the 27th European Conference on Artificial Intelligence (ECAI)},
  year      = {2025}
}
```

---

## Acknowledgements

We sincerely thank the authors of the following previous works for providing their code and templates, which greatly inspired and supported our implementation:

- [**RE-GCN**](https://github.com/Lee-zix/RE-GCN) — Li et al., *Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning* (SIGIR 2021).
- [**DiffuTKG**](https://github.com/AONE-NLP/DiffuTKG) — Diffusion-based framework for Temporal Knowledge Graph reasoning.

Our code is built upon their excellent open-source contributions.