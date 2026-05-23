#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'

"""
Natural Language Inference (NLI) for Semantic Textual Relatedness.

Reduces STR task to recognizing textual entailment using an off-the-shelf NLI classifier.
Computes entailment probability in both directions and averages or takes max.

Model: ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
- Trained on SNLI, MNLI, FEVER, ANLI datasets
- Outputs: entailment, neutral, contradiction probabilities

Bidirectional approach:
- Score(A, B) = combine(P(entailment|A→B), P(entailment|B→A))
- Two modes: average or max

Usage:
  python nli.py --track a --tgt_lan eng --seed 0

Paper Performance: ~61-63% Spearman (not included in final ensemble)
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from config import Config
from src.utils import eva, helper


# NLI Model configuration
NLI_MODEL = 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'

# Weight configuration: [entailment, neutral, contradiction]
# Default: only entailment matters
DEFAULT_WEIGHTS = [1, 0, 0]

# Aggregation mode: 'avg' or 'max'
AGGREGATION_MODE = 'avg'  # Paper used both, avg for dev, max for test


def parse_args():
    """Parse command-line arguments for NLI inference."""
    parser = argparse.ArgumentParser(description='NLI for STR task')
    parser.add_argument('--track', type=str, default='a', help='Track: a, b, c, d, or sts')
    parser.add_argument('--tgt_lan', type=str, default='eng', help='Target language')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--mode', type=str, default='avg', choices=['avg', 'max'],
                       help='Aggregation mode: average or max of bidirectional scores')
    parser.add_argument('--weights', type=str, default='1,0,0',
                       help='Weights for [entailment, neutral, contradiction]')
    return parser.parse_args()


def parse_weights(weights_str):
    """
    Parse weight string to list.

    Args:
        weights_str: Comma-separated weights, e.g., '1,0,0'

    Returns:
        List of float weights
    """
    return [float(w) for w in weights_str.split(',')]


def get_nli_scores(xs1, xs2, model_name=NLI_MODEL, weights=None, mode='avg'):
    """
    Get NLI-based relatedness scores for sentence pairs.

    Uses an NLI classifier to compute entailment probability in both directions.
    Bidirectional: P(entailment|x1→x2) and P(entailment|x2→x1)

    Args:
        xs1: List of first sentences
        xs2: List of second sentences
        model_name: HuggingFace model ID for NLI task
        weights: List of [entailment, neutral, contradiction] weights (default: [1, 0, 0])
        mode: 'avg' to average bidirectional scores, 'max' to take maximum

    Returns:
        List of relatedness scores in range [0, 1]
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    print(f'Loading NLI classifier: {model_name}')
    clf = pipeline('text-classification', model=model_name, return_all_scores=True)

    print(f'Computing NLI scores ({len(xs1)} pairs)...')
    scores = []

    for x1, x2 in tqdm(zip(xs1, xs2), total=len(xs1)):
        # Get NLI output in both directions
        # Output: [{'label': 'entailment', 'score': 0.9}, ...]
        result_1to2 = clf({'text': x1, 'text_pair': x2})  # x1 → x2
        result_2to1 = clf({'text': x2, 'text_pair': x1})  # x2 → x1

        # Extract scores in order: entailment, neutral, contradiction
        # NLI models output in this order
        scores_1to2 = [r['score'] for r in result_1to2]
        scores_2to1 = [r['score'] for r in result_2to1]

        # Apply weights (default: [1, 0, 0] = only entailment)
        weighted_1to2 = sum(s * w for s, w in zip(scores_1to2, weights))
        weighted_2to1 = sum(s * w for s, w in zip(scores_2to1, weights))

        # Aggregate bidirectional scores
        if mode == 'max':
            score = max(weighted_1to2, weighted_2to1)
        else:  # avg
            score = (weighted_1to2 + weighted_2to1) / 2

        scores.append(score)

    return scores


def infer_nli(config, mode='avg', weights=None):
    """
    Main NLI inference workflow.

    Loads dev/test data, computes NLI scores, and saves predictions.

    Args:
        config: Config object with track, tgt_lan, seed settings
        mode: 'avg' or 'max' aggregation mode
        weights: Weight configuration for [entailment, neutral, contradiction]
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    print('\n' + '='*60)
    print('Natural Language Inference (NLI) for STR')
    print('='*60)

    print(f'\nConfiguration:')
    print(f'  Track: {config.track}')
    print(f'  Language: {config.tgt_lan}')
    print(f'  Seed: {config.seed}')
    print(f'  Model: {NLI_MODEL}')
    print(f'  Aggregation mode: {mode}')
    print(f'  Weights [ent, neu, con]: {weights}')

    # Load dev data
    print(f'\nLoading dev data...')
    dev_df = pd.read_csv(config.DEV_LABEL_CSV)
    dev_xs1, dev_xs2 = map(list, zip(*[tuple(row['Text'].split('\n')) for idx, row in dev_df.iterrows()]))
    dev_ys = dev_df['Score'].tolist()

    print(f'  Dev set: {len(dev_xs1)} pairs')

    # Compute NLI scores on dev set
    dev_pred_scores = get_nli_scores(dev_xs1, dev_xs2, NLI_MODEL, weights, mode)

    # Evaluate on dev
    dev_spearman = eva.get_spearman_cor(dev_ys, dev_pred_scores)
    print(f'\nDev Spearman: {dev_spearman:.4f}')

    # Save dev predictions
    results_path = os.path.join(config.RESOURCE_PATH, 'results', config.track, config.tgt_lan, 'nli', str(config.seed))
    os.makedirs(results_path, exist_ok=True)

    results_csv = os.path.join(results_path, f'pred_{config.tgt_lan}_{config.track}.csv')
    results_df = pd.DataFrame({
        'PairID': dev_df['PairID'].tolist(),
        'Pred_Score': dev_pred_scores
    })
    results_df.to_csv(results_csv, index=False)
    print(f'\nDev predictions saved to: {results_csv}')

    # Load test data (if available)
    if os.path.exists(config.TEST_LABEL_CSV):
        print(f'\nLoading test data...')
        test_df = pd.read_csv(config.TEST_LABEL_CSV)
        test_xs1, test_xs2 = map(list, zip(*[tuple(row['Text'].split('\n')) for idx, row in test_df.iterrows()]))
        test_ys = test_df['Score'].tolist()

        print(f'  Test set: {len(test_xs1)} pairs')

        # Compute NLI scores on test set
        test_pred_scores = get_nli_scores(test_xs1, test_xs2, NLI_MODEL, weights, mode)

        # Evaluate on test
        test_spearman = eva.get_spearman_cor(test_ys, test_pred_scores)
        print(f'\nTest Spearman: {test_spearman:.4f}')

        # Save test predictions
        test_results_csv = results_csv.replace('pred_', 'test_pred_')
        test_results_df = pd.DataFrame({
            'PairID': test_df['PairID'].tolist(),
            'Pred_Score': test_pred_scores
        })
        test_results_df.to_csv(test_results_csv, index=False)
        print(f'Test predictions saved to: {test_results_csv}')


if __name__ == '__main__':
    args = parse_args()

    config = Config()
    config.update_config(track=args.track, tgt_lan=args.tgt_lan, seed=args.seed)

    helper.init_logger(config)

    weights = parse_weights(args.weights)
    infer_nli(config, mode=args.mode, weights=weights)
