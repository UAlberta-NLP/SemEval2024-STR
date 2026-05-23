# SemEval2024-STR — Tutorial

Simplified educational implementation of SemEval-2024 Task 1: Semantic Textual Relatedness. Demonstrates the core methods for scoring sentence pairs by semantic relatedness. For the full production system (XGB-4Ms ensemble), see [`system/`](../system/).

## Setup

```sh
cd tutorial
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Run via script:
```sh
python main.py
```

Or interactively:
```sh
jupyter lab   # open main.ipynb
```

## Directory

- `res/` — Datasets, model weights, experiment records
- `src/` — Methods, models, trainers, utility functions
- `main.py` — Entry point
- `config.py` — Configuration

## Dependencies

+ Python >= 3.11
+ jupyterlab
+ numpy
+ pandas
