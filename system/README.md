# SemEval2024-STR - System

## Dependencies
Ensure you have the following dependencies installed:
+ python >= 3.11.9
+ scikit-learn >= 1.5.0
+ xgboost >= 2.1.0
+ pandas >= 2.2.2
+ torch >= 2.3.1
+ datasets >= 2.20.0
+ sentence-transformer >= 3.0.1
+ accelerate >= 0.31.0

## Setup

It is recommended to use a virtual environment to manage dependencies. Follow the steps below to set up the environment and install the required packages:

```sh
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Methods

### FT-MPNet
Run the following command to fine-tune MPNet.
```
$ python sbert.py    
seed: 0
track: a
tgt_lan: eng
method: base
model: sentence-transformers/all-mpnet-base-v2
max_length: 156
train_batch_size: 32
learning_rate: 5e-05
weight_decay: 0.001
adam_epsilon: 1e-08
CURR_PATH: ./
RESOURCE_PATH: ./res
DATA_PATH: ./res/data
SCORES_PATH: ./res/scores
TRACK_PATH: ./res/data/a
LAN_PATH: ./res/data/a/eng
TRAIN_CSV: ./res/data/a/eng/eng_train.csv
DEV_CSV: ./res/data/a/eng/eng_dev.csv
DEV_LABEL_CSV: ./res/data/a/eng/eng_dev_with_labels.csv
TEST_CSV: ./res/data/a/eng/eng_test.csv
TEST_LABEL_CSV: ./res/data/a/eng/eng_test_with_labels.csv
LM_PATH: ./res/lm/sentence-transformers/all-mpnet-base-v2
CKPT_PATH: ./res/ckpts/a/eng/base/sentence-transformers/all-mpnet-base-v2/0
LOG_PATH: ./res/log/a/eng/base/0
LOG_TXT: ./res/log/a/eng/base/0/console_log.txt
RESULTS_PATH: ./res/results/a/eng/base/0
RESULTS_CSV: ./res/results/a/eng/base/0/pred_eng_a.csv
RESULTS_ZIP: ./res/results/a/eng/base/0/pred_eng_a.csv.zip
Target Language eng
If MIX False
Train Size 5500 5500 5500
Val Size 250 250 250
  0%|                                                                                            | 1/5472 [00:04<6:31:47,  4.30s/it]
```

### Ensemble Modules

TODO

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com

## BibTex
Please use the following BibTeX entry to cite us:
```
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