# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

1st-place system (Track A English) for SemEval-2024 Task 1: Semantic Textual Relatedness. The task is to score sentence pairs by degree of semantic relatedness across 14 languages. Two sub-projects live side by side: `system/` (the competition submission) and `tutorial/` (a simplified educational version).

## Setup

Both sub-projects are self-contained — install dependencies from within each directory:

```bash
cd system   # or tutorial/
pip install --upgrade pip
pip install -r requirements.txt
```

All scripts must be run from within their sub-project directory (`system/` or `tutorial/`) because paths are constructed relative to `./`.

## Commands

**Run predictions (dev set):**
```bash
cd system
python main.py --track a --tgt_lan eng --method base --seed 0
```

**Fine-tune models (system only):**
```bash
cd system
# Fine-tune MPNet with sentence-transformers
python finetune.py --model_name mpnet --track a --tgt_lan eng --seed 0

# Fine-tune T5 with regression head
python finetune.py --model_name t5 --track a --tgt_lan eng --seed 0

# Fine-tune GPT-2 with regression head
python finetune.py --model_name gpt2 --track a --tgt_lan eng --seed 0
```

**Run tutorial pipeline:**
```bash
cd tutorial
python main.py --track a --tgt_lan eng --method base
# or interactively:
jupyter lab   # open main.ipynb
```

**Key CLI arguments** (both `main.py` scripts):
- `--track`: `a` (supervised), `b` (unsupervised), `c` (crosslingual), `d`, `sts`
- `--tgt_lan`: `eng afr amh arb arq ary esp hau hin ind kin mar pan tel` — not all languages exist in all tracks; check `res/data/` first
- `--method`: 
  - `base` — Dice coefficient baseline (all sub-projects)
  - `sbert` — Fine-tuned MPNet (system only)
  - `pi` — Paraphrase Identification (system only)
  - `t5` — Fine-tuned T5 (system only)
  - `gpt2` — Fine-tuned GPT-2 (system only)
- `--seed`: integer for reproducibility

## Architecture

### Shared pattern (system/ and tutorial/)

Both sub-projects follow the same structure:

```
config.py          → Config class (argparse → all resource paths)
main.py            → SR class: initialize → dev()
src/methods/base.py → Model base class: predict(s1s, s2s) → List[float]
src/utils/helper.py → init_logger, get_model, zip_file
src/utils/eva.py    → get_spearman_cor(labels, preds) → float
```

### Config auto-generates all paths

`Config.update_config()` derives every resource path from `--track`, `--tgt_lan`, `--method`, and `--seed`:

```
res/data/{track}/{language}/{lang}_{split}.csv     # input CSVs
res/ckpts/{track}/{language}/{method}/{model}/{seed}/  # model checkpoints
res/log/{track}/{language}/{method}/{seed}/console_log.txt
res/results/{track}/{language}/{method}/{seed}/pred_{lang}_{track}.csv
```

### Data format

Input CSVs have columns: `PairID`, `Text` (two sentences joined by `\n`), `Score` (float 0–1).  
Output CSVs have columns: `PairID`, `Pred_Score`. Results are also saved as `.zip`.

### Adding a new method

1. Create `system/src/methods/{name}.py` inheriting from `src/methods/base.py:Model`
2. Implement `.predict(s1s: List[str], s2s: List[str]) -> List[float]`
3. Add the method name to `--method` choices in `config.py`
4. Wire it up in `src/utils/helper.py:get_model()`

### Paper's Official System (XGB-4Ms)

**From paper abstract:**
> "Our official submission for non-English languages in Track C, as well as English in Track A, is a regression ensemble system **XGB-4Ms** designed to synthesize the outputs from fine-tuning **T5, GPT2, RoBERTa, and MPNet**."

**Core Methods in Ensemble** (achieving 0.854 Spearman on Track A English):
1. **Base** (Dice coefficient) — `python main.py --method base`
2. **FT-MPNet** (sbert/contrastive) — `python finetune.py --model_name mpnet`
3. **FT-T5** (regression) — `python finetune.py --model_name t5`
4. **FT-GPT2** (regression) — `python finetune.py --model_name gpt2`
5. **FT-RoBERTa** (regression, outperforms T5/GPT2 individually) — `python finetune.py --model_name roberta`
6. **PI** (Paraphrase Identification, optional diversity input) — `python pi.py`

**Ensemble Training:**
```bash
python ensemble.py --track a --tgt_lan eng --seed 0 --methods base,sbert,pi,t5,gpt2,roberta
```
Uses XGBoost (XGB-4Ms) to learn optimal weights: learns to combine T5, GPT2, RoBERTa, and MPNet predictions.

**Full Reproduction Pipeline:**
```bash
python reproduce.py --track a --tgt_lan eng --seed 0
```
Orchestrates: base → sbert (MPNet) → pi → t5 → gpt2 → roberta → ensemble → results (0.854 Spearman expected).

---

### Method Details

#### PI (Paraphrase Identification)
Two-phase approach: trains RoBERTa binary classifier on general paraphrase datasets, then infers on STR task.

**Phase 1 - Training on Paraphrase Data** (standalone):
- Datasets: PIT, QQP, MRPC, PAWS QQP, PAWS Wiki, PARADE
- Data files: `res/data/paraphrase/{pit,qqp,mrpc,paws_qqp,paws_wiki,parade}_train.csv`
- Does NOT use STR task data — purely paraphrase training
- Outputs checkpoint to: `res/ckpts/{track}/{language}/pi/roberta-base/{seed}/lightning_logs/version_0/checkpoints/`

**Phase 2 - Inference on STR Task** (automatic):
- Loads trained checkpoint
- Bidirectional probability averaging: `score = avg(P(paraphrase|x1,x2), P(paraphrase|x2,x1))`
- Outputs predictions following standard format: `res/results/{track}/{language}/pi/{seed}/pred_{lang}_{track}.csv`

Command:
```bash
python pi.py --track a --tgt_lan eng --seed 0
```

**Paper Performance**: ~51% Spearman on English Track A (lower absolute score but stable across tasks, valuable for ensemble diversity)

#### NLI (Natural Language Inference)
Reduces STR to recognizing textual entailment using an off-the-shelf NLI classifier.

**Approach:**
- Model: `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli` (trained on SNLI, MNLI, FEVER, ANLI)
- Bidirectional: Computes entailment probability in both directions
- Aggregation: Average or max of `P(entailment|x1→x2)` and `P(entailment|x2→x1)`
- Weights: Default focuses on entailment probability (ignores neutral and contradiction)

Command:
```bash
python nli.py --track a --tgt_lan eng --seed 0
# Optional: --mode {avg|max} for aggregation strategy
# Optional: --weights "1,0,0" for [entailment, neutral, contradiction] weights
```

**Paper Performance**: ~64% Spearman on English (dev: 61.5%, test: 63.1%) - not included in final ensemble due to underperformance on some languages and weaker results than other methods.

#### AMR (Abstract Meaning Representation)

Reduces STR to graph-based semantic matching using Abstract Meaning Representation parsing.

**Approach:**
- Parser: SPRING API (https://nlp.uniroma1.it/spring/api/text-to-amr)
- Matching: smatch F-score algorithm (https://github.com/mdtux89/amr-evaluation)
- Reference: "Smatch: an Evaluation Metric for Semantic Feature Structures" (Cai & Knight, 2013)
  https://aclanthology.org/P13-2131/
- Bidirectional: Computes AMR graph similarity in both directions

**Setup Required:**
1. Clone smatch: `git clone https://github.com/mdtux89/amr-evaluation.git`
2. Copy smatch module: `cp amr-evaluation/smatch/smatch.py system/src/smatch/`
3. Ensure API access: https://nlp.uniroma1.it/spring/api/

Command:
```bash
python amr.py --track a --tgt_lan eng --seed 0
```

**Status**: Exploratory method (NOT in final ensemble). Graph-based semantics approach explores AMR representations but underperformed compared to embedding and fine-tuning methods.

#### FT-T5 (Fine-tuned T5 with Regression Head)
Regression fine-tuning of T5-base for STR score prediction.

**Hyperparameters**:
- Model: `t5-base`
- Batch size: 20
- Epochs: 16
- Learning rate: 2e-5
- Loss: MSE (regression to float scores 0-1)
- Metric: Spearman correlation (early stopping)

**Data format**: Text pairs concatenated as `"text1      text2"`

Command:
```bash
python finetune.py --model_name t5 --track a --tgt_lan eng --seed 0
python main.py --track a --tgt_lan eng --method t5 --seed 0
```

**Paper Performance**: ~66% Spearman on English Track A

#### FT-GPT2 (Fine-tuned GPT-2 with Regression Head)
Regression fine-tuning of GPT-2 for STR score prediction.

**Hyperparameters**:
- Model: `gpt2`
- Batch size: 24
- Epochs: 24
- Learning rate: 2e-5
- Loss: MSE (regression to float scores 0-1)
- Metric: Spearman correlation (early stopping)
- **Note**: Requires explicit pad_token configuration (`pad_token = eos_token`)

**Data format**: Text pairs concatenated as `"text1      text2"`

Command:
```bash
python finetune.py --model_name gpt2 --track a --tgt_lan eng --seed 0
python main.py --track a --tgt_lan eng --method gpt2 --seed 0
```

**Paper Performance**: ~66% Spearman on English Track A

#### FT-RoBERTa (Fine-tuned RoBERTa with Regression Head)
Regression fine-tuning of RoBERTa-base for STR score prediction.

**Hyperparameters**:
- Model: `roberta-base`
- Batch size: 24
- Epochs: 24
- Learning rate: 2e-5
- Loss: MSE (regression to float scores 0-1)
- Metric: Spearman correlation (early stopping)

**Data format**: Text pairs concatenated as `"text1      text2"`

Command:
```bash
python finetune.py --model_name roberta --track a --tgt_lan eng --seed 0
python main.py --track a --tgt_lan eng --method roberta --seed 0
```

**Paper Performance**: ~83.6% Spearman on English Track A (outperforms both T5 and GPT-2)

**Note**: RoBERTa fine-tuning produces strong individual performance (83.6%) and is a core component of the XGB-4Ms ensemble system in the paper.

### Non-English Fine-tuning

`sbert.py` augments non-English training data with English-translated pairs from `res/data/trans/{lang}2eng_{split}.csv`. The `MIX` flag at the top of `sbert.py` controls whether the dev set is folded into training. This leverages English pre-training to stabilize predictions on languages with less data.

### Ensemble

The paper's best system uses XGBoost ensemble combining predictions from multiple fine-tuned methods:
- Combines: Base, FT-MPNet, PI, FT-T5, FT-GPT2, FT-RoBERTa
- Hyperparameters: colsample_bytree=0.1, learning_rate=0.1, max_depth=8, alpha=0.1, n_estimators=128, early_stopping_rounds=32
- Trains on dev set with 90/10 train/val split for early stopping
- Paper performance: ~85.6% Spearman on Track A English (XGB-4Ms in paper uses T5, GPT2, RoBERTa, MPNet)

Command:
```bash
python ensemble.py --track a --tgt_lan eng --seed 0 --methods base,sbert,pi,t5,gpt2,roberta
```

This trains XGBoost on predictions from all six methods to learn optimal weighted combination.

## Reproducibility

This repository enables full reproducibility: users can train models from scratch and reproduce the task results.

**Data sources:**
- Training data is committed to `system/res/data/{track}/{language}/` for reproducibility
- Non-English models use English-translated pairs from `system/res/data/trans/`
- Users can download pre-trained models from Hugging Face (e.g., `sentence-transformers/all-mpnet-base-v2`)

### Automated Reproduction (Recommended)

The fastest way to reproduce paper results is the automated reproduction script:

```bash
cd system
pip install -r requirements.txt
python reproduce.py --track a --tgt_lan eng --seed 0
```

This automatically trains all six main methods (Base, FT-MPNet, PI, FT-T5, FT-GPT2, FT-RoBERTa), generates predictions, trains the ensemble, and reports evaluation metrics.

**Note**: NLI is available as an optional additional method (not in the main reproduction pipeline, as it was not competitive with other methods in the paper).

### Manual Method-by-Method Workflow

If you prefer to run methods individually:

```bash
cd system
pip install -r requirements.txt

# 1. Baseline method (Dice coefficient)
python main.py --track a --tgt_lan eng --method base --seed 0

# 2. Fine-tune MPNet with contrastive loss
python finetune.py --model_name mpnet --track a --tgt_lan eng --seed 0
python main.py --track a --tgt_lan eng --method sbert --seed 0

# 3. Train PI model (paraphrase classification on general datasets, then infer on STR)
python pi.py --track a --tgt_lan eng --seed 0

# 3b. (Optional) NLI inference for relatedness estimation
python nli.py --track a --tgt_lan eng --seed 0

# 4. Fine-tune T5 for regression
python finetune.py --model_name t5 --track a --tgt_lan eng --seed 0
python main.py --track a --tgt_lan eng --method t5 --seed 0

# 5. Fine-tune GPT-2 for regression
python finetune.py --model_name gpt2 --track a --tgt_lan eng --seed 0
python main.py --track a --tgt_lan eng --method gpt2 --seed 0

# 6. Fine-tune RoBERTa for regression (strongest individual method)
python finetune.py --model_name roberta --track a --tgt_lan eng --seed 0
python main.py --track a --tgt_lan eng --method roberta --seed 0

# 7. Train ensemble combining all methods
python ensemble.py --track a --tgt_lan eng --seed 0 --methods base,sbert,pi,t5,gpt2,roberta

# Results are saved to:
# - res/results/a/eng/base/0/pred_eng_a.csv (baseline)
# - res/results/a/eng/sbert/0/pred_eng_a.csv (fine-tuned MPNet)
# - res/results/a/eng/pi/0/pred_eng_a.csv (paraphrase identification)
# - res/results/a/eng/t5/0/pred_eng_a.csv (fine-tuned T5)
# - res/results/a/eng/gpt2/0/pred_eng_a.csv (fine-tuned GPT-2)
# - res/results/a/eng/roberta/0/pred_eng_a.csv (fine-tuned RoBERTa)
# - res/results/a/eng/ensemble/0/pred_eng_a.csv (final ensemble)
# - All with corresponding .zip files for submission
```

**Evaluation:**
- Use `src/utils/eva.py:get_spearman_cor()` or `get_pearson_cor()` to evaluate predictions
- Compare correlation metrics against labeled dev/test sets
- Expected ensemble Spearman on Track A English: ~0.854

**Hyperparameter Customization**:
- `finetune.py`: Override defaults via CLI:
  ```bash
  python finetune.py --model_name mpnet --batch_size 16 --epochs 64 --learning_rate 1e-5 --track a --tgt_lan eng --seed 0
  ```
- `pi.py`: Modify `BATCH_SIZE`, `LEARNING_RATE`, `MAX_EPOCHS` at module top
- `ensemble.py`: XGBoost hyperparameters in `train_ensemble()` function

## Team Contributions & Backup Folders

The paper is a team effort combining complementary approaches:

**Production System (`system/`):** Consolidation of all team contributions
- Unified reproducible pipeline (reproduce.py)
- Fine-tuning framework (finetune.py, unified across models)
- Core methods: base, sbert/mpnet, pi, t5, gpt2, ensemble
- Optional exploratory methods: nli, trackb, amr

**Backup Folders (Reference & History):**

1. **`backup_hadi/`** (Hadi Sheikhi)
   - Modular framework with all tracks (A/B/C) and all 14 languages
   - Methods: Dice coefficient, sentence transformers, AMR parsing, translation-based approaches
   - Contributions: `src/methods/base.py`, `src/methods/amr.py`, translation utilities

2. **`backup_senyu/`** (Senyu Li)
   - Transformer fine-tuning experiments (T5, GPT-2, BLOOM)
   - Dev-set variant explorations for multilingual scenarios
   - Contributions: hyperparameter tuning for finetune.py, batch-size/epoch selection

3. **`backup_ning/`** (Ning Shi)
   - Integration & final system assembly
   - Notebooks: Ensemble training, PI extraction, submission pipeline
   - Contributions: `pi.py`, `ensemble.py`, `nli.py`, `trackb.py`, orchestration logic

**Extracted Methods** (from backups → system/):
- `amr.py` — Abstract Meaning Representation parsing (Hadi's contribution)
- `translator.py` — Google Cloud Translation utilities for multilingual support (Hadi's contribution)
- All other methods extracted from notebooks into production scripts

## External Resources

**SemEval-2024 Task 1:**
- **Official Task**: [Semantic Textual Relatedness](https://semantic-textual-relatedness.github.io/)
- **Paper**: [UAlberta at SemEval-2024 Task 1](https://aclanthology.org/2024.semeval-1.254)
- **Leaderboard**: [SemEval-2024 Leaderboard](https://codalab.lisn.upsaclay.fr/competitions/16799)
- **Datasets**: [SemRel2024 Collection](https://arxiv.org/abs/2402.08638)

**Methods & Algorithms:**
- **SPRING AMR Parser**: https://nlp.uniroma1.it/spring/
- **Smatch Evaluation Metric** (AMR matching): https://github.com/mdtux89/amr-evaluation
  - Reference: [Smatch: an Evaluation Metric for Semantic Feature Structures](https://aclanthology.org/P13-2131/) (Cai & Knight, 2013)
- **Sentence Transformers**: https://www.sbert.net/
- **Paraphrase Datasets**: PIT, QQP, MRPC, PAWS, PARADE
