# SemEval2024-STR - Competition System

1st-place system (Track A English) for SemEval-2024 Task 1: Semantic Textual Relatedness. This directory contains the production code for fine-tuning models and generating predictions.

## Quick Start

To reproduce the paper's results with a single command:

```sh
cd system
pip install -r requirements.txt
python reproduce.py --track a --tgt_lan eng --seed 0
```

This automatically trains all five methods, generates predictions, trains the XGB-4Ms ensemble, and reports metrics. Expected output: **0.854 Spearman correlation** on Track A English.

## Paper's Official System (XGB-4Ms Ensemble)

The paper's best system combines 4 fine-tuned transformer models using XGBoost:

**Core Methods** (included in reproduce.py):
1. **Base** (Dice coefficient) — Word overlap baseline
2. **FT-MPNet** (sbert) — Fine-tuned sentence transformers with contrastive loss (~84.9% Spearman)
3. **FT-T5** — Fine-tuned T5-base with regression head (~82.3% Spearman)
4. **FT-GPT2** — Fine-tuned GPT-2 with regression head (~82.9% Spearman)
5. **FT-RoBERTa** — Fine-tuned RoBERTa-base with regression head (~83.6% Spearman)

**Ensemble**: XGBoost (XGB-4Ms) learns to combine T5, GPT2, RoBERTa, MPNet predictions
- **Result**: **0.854 Spearman** on Track A English dev set (85.6% on test set)
- **Command**: `python ensemble.py --track a --tgt_lan eng --seed 0 --methods base,sbert,t5,gpt2,roberta`

**Optional/Exploratory Methods** (available but not in paper's official submission):
- **PI** (pi.py) — Paraphrase Identification using RoBERTa trained on paraphrase data (~51% Spearman, requires separate paraphrase datasets)
- **NLI** (nli.py) — Natural Language Inference using RoBERTa-NLI classifier (~64% Spearman, underperformed on some languages)
- **TrackB** (trackb.py) — Unsupervised Track B ensemble combining BERT and RoBERTa
- **AMR** (amr.py) — Abstract Meaning Representation parsing via external API (graph-based semantics exploration)

## Dependencies
Ensure you have the following dependencies installed:
+ python >= 3.11
+ scikit-learn
+ xgboost
+ pandas
+ torch
+ datasets
+ sentence-transformers >= 3.0
+ transformers
+ accelerate
+ lightning
+ torchmetrics
+ sentencepiece

## Setup

It is recommended to use a virtual environment to manage dependencies. Follow the steps below to set up the environment and install the required packages:

```sh
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

### Data and Models

This repository contains production code but **not** the trained model checkpoints or competition data (they are large). To reproduce the paper:

1. **Download task data** from the [official SemEval-2024 Task 1 competition](https://codalab.lisn.upsaclay.fr/competitions/16799)
2. **Download paraphrase datasets** for PI training (PIT, QQP, MRPC, PAWS, PARADE)
3. **Place data** in `res/data/` following the directory structure in `res/README.md`
4. **Run reproduction script** - trained checkpoints will be generated automatically

See [`res/README.md`](res/README.md) for detailed data setup instructions and directory structure.

## Available Methods (Individual Training)

### 1. Base — Dice Coefficient (Word Overlap)
Baseline method using word overlap similarity.

```sh
$ python main.py --track a --tgt_lan eng --method base --seed 0
```
Output: `res/results/a/eng/base/0/pred_eng_a.csv`

### 2. FT-MPNet — Fine-tuned Sentence Transformers
Fine-tunes `sentence-transformers/all-mpnet-base-v2` with contrastive loss on STR data.
**Paper Performance**: ~83% Spearman on English Track A

```sh
$ python finetune.py --model_name mpnet --track a --tgt_lan eng --seed 0
$ python main.py --track a --tgt_lan eng --method sbert --seed 0
```
Output: `res/results/a/eng/sbert/0/pred_eng_a.csv`

### 3. PI — Paraphrase Identification
Two-phase approach: trains RoBERTa binary classifier on general paraphrase datasets (PIT, QQP, MRPC, PAWS, PARADE), then infers on STR task using bidirectional probability averaging.

**Phase 1 - Training** (standalone, no STR data):
```sh
$ python pi.py --track a --tgt_lan eng --seed 0
```
Trains on: `res/data/paraphrase/{pit,qqp,mrpc,paws_qqp,paws_wiki,parade}_train.csv`
Checkpoint: `res/ckpts/a/eng/pi/roberta-base/0/lightning_logs/version_0/checkpoints/`

**Phase 2 - Inference** (automatic):
Loads trained checkpoint and applies to STR task using bidirectional scoring:
```
score = avg(P(paraphrase|text1,text2), P(paraphrase|text2,text1))
```
Output: `res/results/a/eng/pi/0/pred_eng_a.csv`
**Paper Performance**: ~51% Spearman (stable across tasks, useful for ensemble)

### 3b. NLI — Natural Language Inference (Optional)
Reduces STR to recognizing textual entailment using an off-the-shelf RoBERTa NLI classifier trained on SNLI, MNLI, FEVER, ANLI datasets.

**Approach**: Bidirectional entailment probability averaging
```
score = avg(P(entailment|text1→text2), P(entailment|text2→text1))
```

```sh
$ python nli.py --track a --tgt_lan eng --seed 0
```
Output: `res/results/a/eng/nli/0/pred_eng_a.csv`
**Paper Performance**: ~64% Spearman (not included in final ensemble)

**Note**: NLI was tested but underperformed on some languages and was not competitive with other methods, so it was excluded from the best ensemble system.

### 4. FT-T5 — Fine-tuned T5 with Regression Head
Regression fine-tuning of T5-base for STR score prediction.
**Hyperparameters**: batch_size=24, epochs=16, lr=2e-5, MSE loss
**Paper Performance**: ~66% Spearman

```sh
$ python finetune.py --model_name t5 --track a --tgt_lan eng --seed 0
$ python main.py --track a --tgt_lan eng --method t5 --seed 0
```
Output: `res/results/a/eng/t5/0/pred_eng_a.csv`

### 5. FT-GPT2 — Fine-tuned GPT-2 with Regression Head
Regression fine-tuning of GPT-2 for STR score prediction.
**Hyperparameters**: batch_size=24, epochs=24, lr=2e-5, MSE loss
**Paper Performance**: ~66% Spearman

```sh
$ python finetune.py --model_name gpt2 --track a --tgt_lan eng --seed 0
$ python main.py --track a --tgt_lan eng --method gpt2 --seed 0
```
Output: `res/results/a/eng/gpt2/0/pred_eng_a.csv`

## Data Format

**Input CSVs** (`res/data/{track}/{language}/{lang}_{split}.csv`):
```
PairID,Text,Score
1,"text1
text2",0.85
```

**Output CSVs** (`res/results/{track}/{language}/{method}/{seed}/pred_{lang}_{track}.csv`):
```
PairID,Pred_Score
1,0.87
```

### 6. FT-RoBERTa — Fine-tuned RoBERTa with Regression Head
Regression fine-tuning of RoBERTa-base for STR score prediction.
**Hyperparameters**: batch_size=24, epochs=24, lr=2e-5, MSE loss
**Paper Performance**: ~83.6% Spearman (strongest individual method)

```sh
$ python finetune.py --model_name roberta --track a --tgt_lan eng --seed 0
$ python main.py --track a --tgt_lan eng --method roberta --seed 0
```
Output: `res/results/a/eng/roberta/0/pred_eng_a.csv`

### 7. Ensemble — XGBoost Combining All Methods (XGB-4Ms)

The paper's best system combines predictions from all five methods using XGBoost:

```sh
$ python ensemble.py --track a --tgt_lan eng --seed 0 --methods base,sbert,t5,gpt2,roberta
```

**Hyperparameters**:
- Objective: squared error regression
- Learning rate: 0.1
- Max depth: 8
- Column sample: 0.1
- Estimators: 128
- Early stopping: 32 rounds on 10% validation set

**Paper Performance**: **0.854 Spearman** on English Track A (best reported)

Output: `res/results/a/eng/ensemble/0/pred_eng_a.csv`

## Non-English Training

For non-English languages, `finetune.py` (for MPNet) augments training data with English translations from `res/data/trans/{lang}2eng_{split}.csv` to leverage English-specific pre-training.

## Ensemble Strategy

The XGB-4Ms ensemble combines predictions from:
1. **Base** (Dice coefficient): ~41% Spearman — simple baseline
2. **FT-MPNet** (Contrastive): ~84.9% Spearman — strong single method
3. **FT-T5** (Regression): ~82.3% Spearman — general-purpose LLM
4. **FT-GPT2** (Regression): ~82.9% Spearman — autoregressive variant
5. **FT-RoBERTa** (Regression): ~83.6% Spearman — strongest individual method

XGBoost learns to weight these methods optimally, achieving **0.854 Spearman** by leveraging their complementary strengths.

## References

- **Paper**: [UAlberta at SemEval-2024 Task 1](https://aclanthology.org/2024.semeval-1.254)
- **Task**: [SemEval-2024 Task 1: Semantic Textual Relatedness](https://semantic-textual-relatedness.github.io/)
- **Leaderboard**: [SemEval-2024 Leaderboard](https://codalab.lisn.upsaclay.fr/competitions/16799)

## Authors
* Ning Shi - mrshininnnnn@gmail.com

## BibTeX
Please use the following BibTeX entry to cite this work:
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
    abstract = "We describe our systems for SemEval-2024 Task 1: Semantic Textual Relatedness. We investigate the correlation between semantic relatedness and semantic similarity. Specifically, we test two hypotheses: (1) similarity is a special case of relatedness, and (2) semantic relatedness is preserved under translation. We experiment with a variety of approaches which are based on explicit semantics, downstream applications, contextual embeddings, large language models (LLMs), as well as ensembles of methods. We find empirical support for our theoretical insights. In addition, our best ensemble system yields highly competitive results in a number of diverse categories. Our code and data are available on GitHub.",
}
```