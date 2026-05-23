# Resources Directory

This directory contains data, model checkpoints, and results for the SemEval-2024 Task 1 system.

## Directory Structure

```
res/
├── data/           # Input datasets (training, dev, test)
├── ckpts/          # Model checkpoints (downloaded or trained)
├── lm/             # Language model weights (downloaded)
├── results/        # Prediction outputs (generated)
├── log/            # Training logs (generated)
└── scores/         # Reference scores (reference only)
```

## Data Setup

### Download Training Data

The task data should be downloaded from the official SemEval-2024 Task 1 competition page:
- **Official Task**: https://semantic-textual-relatedness.github.io/
- **Download Link**: https://codalab.lisn.upsaclay.fr/competitions/16799

Place downloaded data in `data/{track}/{language}/`:

```bash
res/data/
├── a/                          # Track A (Supervised)
│   ├── eng/                    # English
│   │   ├── eng_train.csv       # Training data
│   │   ├── eng_dev.csv         # Dev data (no labels)
│   │   ├── eng_dev_with_labels.csv    # Dev with labels
│   │   ├── eng_test.csv        # Test data
│   │   └── eng_test_with_labels.csv   # Test with labels
│   ├── tel/                    # Telugu
│   ├── esp/                    # Spanish
│   └── ...                     # Other languages
├── b/                          # Track B (Unsupervised)
├── c/                          # Track C (Cross-lingual)
├── d/                          # Track D
└── paraphrase/                 # Paraphrase datasets for PI training
    ├── pit_train.csv
    ├── qqp_train.csv
    ├── mrpc_train.csv
    ├── paws_qqp_train.csv
    ├── paws_wiki_train.csv
    └── parade_train.csv
```

### Paraphrase Datasets for PI Training

The PI method requires paraphrase datasets. Download from:
- **PIT** (Paraphrase in Twitter): https://www.aclweb.org/anthology/P15-1009
- **QQP** (Quora Question Pairs): https://www.quora.com/q/quoradata
- **MRPC** (Microsoft Research Paraphrase Corpus): https://www.microsoft.com/en-us/download/details.aspx?id=52398
- **PAWS** (Paraphrase Adversaries from Word Scrambling): https://github.com/google-research-datasets/paws
- **PARADE**: https://github.com/guydobash/PARADE

Place in `data/paraphrase/` with naming convention: `{dataset}_train.csv`

## Model Checkpoints

### Pre-trained Models (Auto-downloaded)

The following models are automatically downloaded by HuggingFace transformers:

**Fine-tuning Models:**
- `sentence-transformers/all-mpnet-base-v2` - Used for FT-MPNet
- `t5-base` - Used for FT-T5
- `gpt2` - Used for FT-GPT2
- `roberta-base` - Used for FT-RoBERTa (core XGB-4Ms) and PI training

**Inference Models:**
- `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli` - Used for NLI inference

Models are cached in HuggingFace's default location (usually `~/.cache/huggingface/`).

### Trained Checkpoints (Generated)

After running the training scripts, checkpoints are saved to:

```
ckpts/
├── a/                    # Track A checkpoints
│   ├── eng/
│   │   ├── sbert/        # FT-MPNet checkpoints
│   │   │   └── 0/        # seed=0
│   │   ├── t5/           # FT-T5 checkpoints
│   │   │   └── 0/
│   │   ├── gpt2/         # FT-GPT2 checkpoints
│   │   │   └── 0/
│   │   ├── roberta/      # FT-RoBERTa checkpoints (core XGB-4Ms)
│   │   │   └── 0/
│   │   ├── pi/           # PI checkpoints (RoBERTa paraphrase)
│   │   │   └── roberta-base/
│   │   │       └── 0/
│   │   └── ensemble/     # Ensemble (XGBoost)
│   │       └── 0/
│   └── ...               # Other languages
├── b/                    # Track B checkpoints
├── c/                    # Track C checkpoints
└── ...
```

## Quick Start for Data Setup

```bash
cd system

# 1. Install dependencies
pip install -r requirements.txt

# 2. Download SemEval-2024 Task 1 data from:
#    https://codalab.lisn.upsaclay.fr/competitions/16799
#    Extract to: res/data/

# 3. Download paraphrase datasets and extract to: res/data/paraphrase/
#    (See links above)

# 4. Run reproduction (this will train all models and generate checkpoints)
python reproduce.py --track a --tgt_lan eng --seed 0

# 5. Results will be saved to: res/results/
```

## File Format

### Input CSVs (Training/Dev/Test Data)

```
PairID,Text,Score
1,"sentence1
sentence2",0.85
2,"sentence1
sentence2",0.42
```

**Columns:**
- `PairID`: Unique identifier for the sentence pair
- `Text`: Two sentences separated by newline (`\n`)
- `Score`: Relatedness score (0.0 to 1.0) - only in labeled datasets

### Output CSVs (Predictions)

```
PairID,Pred_Score
1,0.87
2,0.39
```

**Columns:**
- `PairID`: Matches input
- `Pred_Score`: Predicted relatedness score (rounded to 2 decimals)

## Disk Space Requirements

Approximate space needed:

```
res/data/          ~1-2 GB   (SemEval data + paraphrase datasets)
res/ckpts/         ~15-20 GB (All trained model checkpoints)
res/lm/            ~10-15 GB (Downloaded pre-trained models)
res/results/       ~100-200 MB (Prediction outputs)
```

**Total: ~27-37 GB** for full reproduction with all languages and checkpoints.

## Non-English Language Setup

For non-English languages (Track C, D), translated data is used:

```
res/data/trans/
├── tel2eng_train.csv      # Telugu translated to English
├── tel2eng_dev.csv
├── esp2eng_train.csv      # Spanish translated to English
├── esp2eng_dev.csv
└── ...
```

These translations are typically from Google Translate API (applied during training).

