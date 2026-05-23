#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'

"""
Track B Unsupervised Ensemble for Semantic Textual Relatedness.

Linear ensemble combining BERT and RoBERTa predictions for Track B (unsupervised).
Uses min-max normalization before averaging to handle different score ranges.

Usage:
  python trackb.py --track b --tgt_lan eng --seed 0
"""

import os
import argparse
import numpy as np
import pandas as pd

from config import Config
from src.utils import helper


def normalize_list(lst):
    """
    Min-max normalize a list of values to [0, 1].

    Handles edge case where all values are identical (returns zeros).

    Args:
        lst: List of numeric values to normalize

    Returns:
        List of normalized values in range [0, 1]
    """
    min_val = min(lst)
    max_val = max(lst)
    range_val = max_val - min_val

    if range_val == 0:
        return [0.0 for _ in lst]

    return [(item - min_val) / range_val for item in lst]


def load_method_predictions(config, method_name, split):
    """
    Load predictions from a specific method for a given split.

    Args:
        config: Config object with track and language settings
        method_name: Name of method ('bert', 'roberta', etc.)
        split: Data split ('dev' or 'test')

    Returns:
        List of prediction scores
    """
    csv_path = os.path.join(
        config.RESOURCE_PATH, 'str', split, config.tgt_lan, f'{method_name}.csv'
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Predictions not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df['Pred_Score'].tolist()


def ensemble_trackb(config, methods=None):
    """
    Main Track B unsupervised ensemble workflow.

    Combines multiple method predictions using min-max normalization and averaging.

    Args:
        config: Config object with track, tgt_lan, seed settings
        methods: List of method names to ensemble (default: ['bert', 'roberta'])
    """
    if methods is None:
        methods = ['bert', 'roberta']

    print('\n' + '='*60)
    print('Track B Unsupervised Ensemble')
    print('='*60)

    print(f'\nConfiguration:')
    print(f'  Track: {config.track}')
    print(f'  Language: {config.tgt_lan}')
    print(f'  Seed: {config.seed}')
    print(f'  Methods: {", ".join(methods)}')

    # Load dev data to match PairIDs
    print(f'\nLoading dev data...')
    dev_df = pd.read_csv(config.DEV_CSV)
    print(f'  Dev pairs: {len(dev_df)}')

    # Ensemble dev predictions
    print(f'\nEnsembling dev predictions...')
    dev_scores = []
    for method in methods:
        try:
            predictions = load_method_predictions(config, method, 'dev')
            normalized = normalize_list(predictions)
            dev_scores.append(normalized)
            print(f'  {method}: loaded {len(predictions)} predictions')
        except FileNotFoundError as e:
            print(f'  Warning: {e}')

    if not dev_scores:
        raise RuntimeError('No method predictions loaded for dev set')

    dev_scores = np.array(dev_scores)
    dev_ensemble = np.mean(dev_scores, axis=0).tolist()

    # Save dev predictions
    dev_results_path = os.path.join(
        config.RESOURCE_PATH, 'results', config.track, config.tgt_lan, 'trackb', str(config.seed)
    )
    os.makedirs(dev_results_path, exist_ok=True)

    dev_csv = os.path.join(dev_results_path, f'pred_{config.tgt_lan}_{config.track}.csv')
    dev_results_df = pd.DataFrame({
        'PairID': dev_df['PairID'].tolist(),
        'Pred_Score': dev_ensemble
    })
    dev_results_df.to_csv(dev_csv, index=False)
    print(f'Dev predictions saved to: {dev_csv}')

    # Load and ensemble test predictions (if available)
    if os.path.exists(config.TEST_CSV):
        print(f'\nLoading test data...')
        test_df = pd.read_csv(config.TEST_CSV)
        print(f'  Test pairs: {len(test_df)}')

        print(f'\nEnsembling test predictions...')
        test_scores = []
        for method in methods:
            try:
                predictions = load_method_predictions(config, method, 'test')
                normalized = normalize_list(predictions)
                test_scores.append(normalized)
                print(f'  {method}: loaded {len(predictions)} predictions')
            except FileNotFoundError as e:
                print(f'  Warning: {e}')

        if test_scores:
            test_scores = np.array(test_scores)
            test_ensemble = np.mean(test_scores, axis=0).tolist()

            test_csv = os.path.join(dev_results_path, f'test_pred_{config.tgt_lan}_{config.track}.csv')
            test_results_df = pd.DataFrame({
                'PairID': test_df['PairID'].tolist(),
                'Pred_Score': test_ensemble
            })
            test_results_df.to_csv(test_csv, index=False)
            print(f'Test predictions saved to: {test_csv}')


def parse_args():
    """Parse command-line arguments for Track B ensemble."""
    parser = argparse.ArgumentParser(description='Track B unsupervised ensemble')
    parser.add_argument('--track', type=str, default='b', help='Track: b')
    parser.add_argument('--tgt_lan', type=str, default='eng', help='Target language')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--methods', type=str, default='bert,roberta',
                       help='Comma-separated list of methods to ensemble')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = Config()
    config.update_config(track=args.track, tgt_lan=args.tgt_lan, seed=args.seed)

    helper.init_logger(config)

    methods = [m.strip() for m in args.methods.split(',')]
    ensemble_trackb(config, methods=methods)
