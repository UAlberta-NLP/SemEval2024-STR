# SemEval2024-STR — Competition System

Production code for the 1st-place system (Track A English) for SemEval-2024 Task 1. See the [root README](../README.md) for project overview, citation, and quick start.

## Setup

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

All scripts must be run from within this `system/` directory.

## Reproduce

```sh
python reproduce.py --track a --tgt_lan eng --seed 0
```

Trains all five methods, generates predictions, trains the XGB-4Ms ensemble, and reports metrics. Expected: **0.854 Spearman** on Track A English dev set.

## Data

This repository contains training data in `res/data/` but not trained checkpoints (they are large). See [`res/README.md`](res/README.md) for directory structure and download instructions.

## Dependencies

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

## Paper's Official System (XGB-4Ms)

**Core Methods** (run by `reproduce.py`):

| # | Method | Model | Performance |
|---|--------|-------|-------------|
| 1 | Base | Dice coefficient | ~41% Spearman |
| 2 | FT-MPNet | sentence-transformers/all-mpnet-base-v2 | ~84.9% Spearman |
| 3 | FT-T5 | T5-base (regression) | ~82.3% Spearman |
| 4 | FT-GPT2 | GPT-2 (regression) | ~82.9% Spearman |
| 5 | FT-RoBERTa | RoBERTa-base (regression) | ~83.6% Spearman |
| — | **XGB-4Ms Ensemble** | XGBoost combining all | **85.4% Spearman** |

**Optional/Exploratory Methods** (not in paper's official submission):
- **PI** (`pi.py`) — Paraphrase Identification (~51% Spearman, requires separate paraphrase datasets)
- **NLI** (`nli.py`) — Natural Language Inference (~64% Spearman)
- **AMR** (`amr.py`) — Abstract Meaning Representation via external API
- **TrackB** (`trackb.py`) — Unsupervised Track B ensemble

## Available Methods

### 1. Base — Dice Coefficient

```sh
python main.py --track a --tgt_lan eng --method base --seed 0
```
Output: `res/results/a/eng/base/0/pred_eng_a.csv`

### 2. FT-MPNet — Fine-tuned Sentence Transformers

Fine-tunes `sentence-transformers/all-mpnet-base-v2` with contrastive loss.
**Hyperparameters**: batch_size=32, epochs=32, lr=5e-5

```sh
python finetune.py --model_name mpnet --track a --tgt_lan eng --seed 0
python main.py --track a --tgt_lan eng --method sbert --seed 0
```
Output: `res/results/a/eng/sbert/0/pred_eng_a.csv`

### 3. FT-T5 — Fine-tuned T5 with Regression Head

**Hyperparameters**: batch_size=24, epochs=16, lr=2e-5, MSE loss

```sh
python finetune.py --model_name t5 --track a --tgt_lan eng --seed 0
python main.py --track a --tgt_lan eng --method t5 --seed 0
```
Output: `res/results/a/eng/t5/0/pred_eng_a.csv`

### 4. FT-GPT2 — Fine-tuned GPT-2 with Regression Head

**Hyperparameters**: batch_size=24, epochs=24, lr=2e-5, MSE loss

```sh
python finetune.py --model_name gpt2 --track a --tgt_lan eng --seed 0
python main.py --track a --tgt_lan eng --method gpt2 --seed 0
```
Output: `res/results/a/eng/gpt2/0/pred_eng_a.csv`

### 5. FT-RoBERTa — Fine-tuned RoBERTa with Regression Head

Strongest individual method. **Hyperparameters**: batch_size=24, epochs=24, lr=2e-5, MSE loss

```sh
python finetune.py --model_name roberta --track a --tgt_lan eng --seed 0
python main.py --track a --tgt_lan eng --method roberta --seed 0
```
Output: `res/results/a/eng/roberta/0/pred_eng_a.csv`

### 6. XGB-4Ms Ensemble

Combines predictions from all five methods using XGBoost.

```sh
python ensemble.py --track a --tgt_lan eng --seed 0 --methods base,sbert,t5,gpt2,roberta
```

**XGBoost Hyperparameters**: objective=reg:squarederror, learning_rate=0.1, max_depth=8, colsample_bytree=0.1, n_estimators=128, early_stopping_rounds=32 (on 10% val split)

Output: `res/results/a/eng/ensemble/0/pred_eng_a.csv`

## Data Format

**Input** (`res/data/{track}/{language}/{lang}_{split}.csv`):
```
PairID,Text,Score
1,"sentence1
sentence2",0.85
```

**Output** (`res/results/{track}/{language}/{method}/{seed}/pred_{lang}_{track}.csv`):
```
PairID,Pred_Score
1,0.87
```

## Non-English Training

For non-English languages, `finetune.py` augments training data with English translations from `res/data/trans/{lang}2eng_{split}.csv`. The `MIX` flag (default `False`) at the top of `finetune.py` controls whether the dev set is folded into training.
