#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'

"""
XGBoost Ensemble for combining predictions from multiple methods.

Trains an XGBoost regressor on stacked predictions from individual methods
to generate an ensemble prediction. Each method's prediction becomes a feature,
and the ensemble learns weighted combinations.

Usage:
  python ensemble.py --track a --tgt_lan eng --seed 0 --methods base,sbert,pi,t5,gpt2
"""

import os
import argparse
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from config import Config
from src.utils import eva


def parse_args():
    """Parse command-line arguments for ensemble training."""
    parser = argparse.ArgumentParser(description='XGBoost ensemble for STR predictions')
    parser.add_argument('--track', type=str, default='a', help='Track: a, b, c, d, or sts')
    parser.add_argument('--tgt_lan', type=str, default='eng', help='Target language')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--methods', type=str, default='base,sbert,pi,t5,gpt2',
                       help='Comma-separated list of methods to ensemble')
    return parser.parse_args()


def load_predictions(csv_path):
    """
    Load predictions from a CSV file.

    Args:
        csv_path: Path to CSV with columns [PairID, Pred_Score]

    Returns:
        numpy array of shape (num_samples,) containing predictions
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prediction file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df['Pred_Score'].values


def load_ground_truth(csv_path):
    """
    Load ground truth labels from a CSV file.

    Args:
        csv_path: Path to CSV with columns [PairID, Text, Score]

    Returns:
        numpy array of shape (num_samples,) containing scores
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Label file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df['Score'].values


def train_ensemble(X_train, y_train, X_val, y_val, seed=0):
    """
    Train XGBoost ensemble on stacked predictions.

    Args:
        X_train: Feature matrix of shape (num_train, num_methods) with method predictions
        y_train: Ground truth labels of shape (num_train,)
        X_val: Validation feature matrix of shape (num_val, num_methods)
        y_val: Validation labels of shape (num_val,)
        seed: Random seed for reproducibility

    Returns:
        Trained XGBRegressor model
    """
    model = XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.1,
        learning_rate=0.1,
        max_depth=8,
        alpha=0.1,
        n_estimators=128,
        early_stopping_rounds=32,
        random_state=seed,
        verbose=False
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def ensemble_method(config, methods, seed=0):
    """
    Main ensemble workflow: load predictions, train ensemble, generate results.

    Args:
        config: Config object with track, tgt_lan, seed settings
        methods: List of method names to ensemble (e.g., ['base', 'sbert', 'pi', 't5', 'gpt2'])
        seed: Random seed
    """
    print('\n' + '='*60)
    print('XGBoost Ensemble for STR Predictions')
    print('='*60)

    print(f'\nConfiguration:')
    print(f'  Track: {config.track}')
    print(f'  Language: {config.tgt_lan}')
    print(f'  Methods: {", ".join(methods)}')
    print(f'  Seed: {seed}')

    # Load training predictions from all methods
    print(f'\nLoading training predictions from {len(methods)} methods...')
    train_predictions = []
    for method in methods:
        method_config = Config()
        method_config.update_config(track=config.track, tgt_lan=config.tgt_lan, method=method, seed=seed)
        pred_path = method_config.RESULTS_CSV

        if not os.path.exists(pred_path):
            print(f'  WARNING: {method} predictions not found at {pred_path}')
            continue

        preds = load_predictions(pred_path)
        train_predictions.append(preds)
        print(f'  {method}: {len(preds)} samples loaded')

    if not train_predictions:
        raise ValueError('No method predictions found. Please run individual methods first.')

    # Stack predictions: shape (num_methods, num_samples) -> (num_samples, num_methods)
    X_train = np.vstack(train_predictions).T
    print(f'\nStacked training features: shape {X_train.shape}')

    # Load ground truth labels from dev set
    y_train = load_ground_truth(config.DEV_LABEL_CSV)
    print(f'Ground truth labels: shape {y_train.shape}')

    # Train/val split: 90/10
    num_samples = len(y_train)
    num_train = int(0.9 * num_samples)

    X_train_split = X_train[:num_train]
    y_train_split = y_train[:num_train]
    X_val_split = X_train[num_train:]
    y_val_split = y_train[num_train:]

    print(f'\nTrain/val split:')
    print(f'  Train: {len(X_train_split)} samples')
    print(f'  Val: {len(X_val_split)} samples')

    # Train ensemble
    print(f'\nTraining XGBoost ensemble...')
    model = train_ensemble(X_train_split, y_train_split, X_val_split, y_val_split, seed=seed)

    # Evaluate on validation set
    y_val_pred = model.predict(X_val_split)
    val_spearman = eva.get_spearman_cor(y_val_split, y_val_pred)
    print(f'  Validation Spearman: {val_spearman:.4f}')

    # Generate predictions on full dev set
    y_pred = model.predict(X_train)
    dev_spearman = eva.get_spearman_cor(y_train, y_pred)
    print(f'  Dev Spearman: {dev_spearman:.4f}')

    # Round predictions to 2 decimals and clip to [0, 1]
    y_pred = np.clip(np.round(y_pred, 2), 0, 1)

    # Save results
    results_path = os.path.join(config.RESOURCE_PATH, 'results', config.track, config.tgt_lan, 'ensemble', str(seed))
    os.makedirs(results_path, exist_ok=True)

    results_csv = os.path.join(results_path, f'pred_{config.tgt_lan}_{config.track}.csv')
    results_df = pd.DataFrame({
        'PairID': range(1, len(y_pred) + 1),
        'Pred_Score': y_pred
    })
    results_df.to_csv(results_csv, index=False)
    print(f'\nResults saved to: {results_csv}')

    # Feature importance
    print(f'\nFeature importance:')
    for i, method in enumerate(methods):
        importance = model.feature_importances_[i]
        print(f'  {method}: {importance:.4f}')

    return model, results_df


if __name__ == '__main__':
    args = parse_args()

    config = Config()
    config.update_config(track=args.track, tgt_lan=args.tgt_lan, seed=args.seed)

    methods = args.methods.split(',')
    ensemble_method(config, methods, seed=args.seed)
