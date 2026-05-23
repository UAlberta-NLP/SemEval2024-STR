# UAlberta at SemEval-2024 Task 1: Semantic Textual Relatedness

[![Paper](https://img.shields.io/badge/Paper-ACL%20Anthology-red?style=flat-square)](https://aclanthology.org/2024.semeval-1.254/)
[![Poster](https://img.shields.io/badge/Poster-PDF-blue?style=flat-square)](assets/poster.pdf)
[![Slides](https://img.shields.io/badge/Slides-PDF-green?style=flat-square)](assets/slides.pdf)

This repository contains the **1st-place system** for SemEval-2024 Task 1: Semantic Textual Relatedness (Track A English). We investigate semantic relatedness across 14 languages using a diverse ensemble of methods combining explicit semantics, downstream applications, contextual embeddings, and large language models.

:trophy: **1st Place on Track A English** | **0.856 Spearman Correlation**

## Quick Start

To reproduce the paper's results:

```bash
cd system
pip install -r requirements.txt
python reproduce.py --track a --tgt_lan eng --seed 0
```

This automatically trains the XGB-4Ms ensemble (T5, GPT-2, RoBERTa, MPNet) and reports **0.854 Spearman** on the dev set.

## System Overview

The paper's best system (**XGB-4Ms**) combines predictions from 4 fine-tuned transformer models using XGBoost:

| Model | Architecture | Performance |
|-------|--------------|-------------|
| **FT-MPNet** | Sentence-Transformers (contrastive) | 84.9% |
| **FT-RoBERTa** | RoBERTa-base (regression) | 83.6% |
| **FT-GPT2** | GPT-2 (regression) | 82.9% |
| **FT-T5** | T5-base (regression) | 82.3% |
| **Ensemble** | XGBoost (XGB-4Ms) | **85.6%** |

Additional methods explored: PI (Paraphrase Identification), NLI (Natural Language Inference), AMR (Abstract Meaning Representation).

## Directory Structure

```
.
├── system/                 # Production system (ready to reproduce)
│   ├── reproduce.py       # Orchestrates full pipeline
│   ├── finetune.py        # Fine-tunes all models (mpnet, t5, gpt2, roberta)
│   ├── ensemble.py        # XGBoost ensemble combining methods
│   ├── pi.py              # Paraphrase identification (RoBERTa)
│   ├── nli.py             # Natural language inference (optional)
│   ├── amr.py             # Abstract meaning representation (optional)
│   ├── base.py            # Dice coefficient baseline
│   ├── main.py            # Main entry point for predictions
│   ├── config.py          # Configuration management
│   ├── requirements.txt    # Dependencies
│   └── res/               # Data, checkpoints, results (auto-generated)
├── tutorial/              # Educational implementation (simplified)
├── README.md              # This file
└── assets/                # Paper, poster, slides

```

## Usage

### Full Reproduction (Recommended)
```bash
cd system
python reproduce.py --track a --tgt_lan eng --seed 0
```
Trains all methods, generates predictions, and reports metrics.

### Individual Methods
```bash
# Baseline (Dice coefficient)
python main.py --track a --tgt_lan eng --method base --seed 0

# Fine-tune specific models
python finetune.py --model_name mpnet --track a --tgt_lan eng --seed 0
python finetune.py --model_name t5 --track a --tgt_lan eng --seed 0
python finetune.py --model_name gpt2 --track a --tgt_lan eng --seed 0
python finetune.py --model_name roberta --track a --tgt_lan eng --seed 0

# Optional methods
python pi.py --track a --tgt_lan eng --seed 0          # Paraphrase ID (optional)
python nli.py --track a --tgt_lan eng --seed 0         # NLI (optional)

# Ensemble (XGB-4Ms)
python ensemble.py --track a --tgt_lan eng --seed 0 --methods base,sbert,t5,gpt2,roberta
```

For detailed documentation on all methods and configuration, see `system/README.md`.

## Requirements

- Python >= 3.11
- PyTorch
- Transformers
- Sentence-Transformers >= 3.0
- XGBoost

Full dependencies in `system/requirements.txt`.

## Data & Models

**Input Data**: Download from [SemEval-2024 Task 1 Competition](https://codalab.lisn.upsaclay.fr/competitions/16799)

**Pre-trained Models**: Automatically downloaded from HuggingFace:
- sentence-transformers/all-mpnet-base-v2
- t5-base
- gpt2
- roberta-base

See `system/res/README.md` for detailed setup instructions.

## Author
Ning Shi — mrshininnnnn@gmail.com

## Citation

```bibtex
@inproceedings{shi-etal-2024-ualberta,
    title = "{UA}lberta at {S}em{E}val-2024 Task 1: A Potpourri of Methods for Quantifying Multilingual Semantic Textual Relatedness and Similarity",
    author = "Shi, Ning  and
      Li, Senyu  and
      Luo, Guoqing  and
      Mirzaei, Amirreza  and
      Rafiei, Ali  and
      Riley, Jai  and
      Sheikhi, Hadi  and
      Siavashpour, Mahvash  and
      Tavakoli, Mohammad  and
      Hauer, Bradley",
    editor = {Ojha, Atul Kr.  and
      Do{\u{g}}ru{\"o}z, A. Seza  and
      Tayyar Madabushi, Harish  and
      Da San Martino, Giovanni  and
      Rosenthal, Sara  and
      Ros{\'a}, Aiala},
    booktitle = "Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.semeval-1.254",
    pages = "1798--1805",
}
```

## References

- **Task**: [Semantic Textual Relatedness](https://semantic-textual-relatedness.github.io/)
- **Dataset**: [SemRel2024 Collection](https://arxiv.org/abs/2402.08638)
- **Leaderboard**: [SemEval-2024 Leaderboard](https://codalab.lisn.upsaclay.fr/competitions/16799)