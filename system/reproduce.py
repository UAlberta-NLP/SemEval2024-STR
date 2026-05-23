#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'

"""
Automated reproduction script for SemEval-2024 Task 1: Semantic Textual Relatedness.

Trains five methods in sequence (Base, Fine-tuned MPNet, Fine-tuned T5, Fine-tuned GPT-2,
Fine-tuned RoBERTa), generates predictions for each, trains an XGBoost ensemble (XGB-4Ms)
combining all methods, and outputs final results with evaluation metrics.

This is the main entry point for reproducing the paper's results.

Usage:
  python reproduce.py --track a --tgt_lan eng --seed 0

Output:
  - Individual method predictions: res/results/{track}/{language}/{method}/{seed}/
  - Ensemble predictions: res/results/{track}/{language}/ensemble/{seed}/
  - Evaluation metrics printed to console
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np

from config import Config
from src.utils import eva


def parse_args():
    """Parse command-line arguments for reproduction workflow."""
    parser = argparse.ArgumentParser(description='Reproduce SemEval-2024 Task 1 results')
    parser.add_argument('--track', type=str, default='a', help='Track: a, b, c, d, or sts')
    parser.add_argument('--tgt_lan', type=str, default='eng', help='Target language')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--skip_methods', action='store_true', help='Skip method training, only run ensemble')
    return parser.parse_args()


def ckpt_exists(track, tgt_lan, method, model_id, seed):
    """Check if a checkpoint already exists for this method."""
    model_name = model_id.replace('/', '_')
    ckpt_path = os.path.join('res', 'ckpts', track, tgt_lan, method, model_name, str(seed))
    return os.path.isdir(ckpt_path) and bool(os.listdir(ckpt_path))


def result_exists(track, tgt_lan, method, seed):
    """Check if prediction results already exist for this method."""
    result_path = os.path.join('res', 'results', track, tgt_lan, method, str(seed), f'pred_{tgt_lan}_{track}.csv')
    return os.path.exists(result_path)


def run_command(cmd, description):
    """
    Execute a shell command and report progress.

    Args:
        cmd: Shell command as string
        description: Human-readable description of the command

    Returns:
        True if successful, False otherwise
    """
    print(f'\n{"="*70}')
    print(f'{description}')
    print('='*70)
    print(f'Command: {cmd}\n')

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f'\nERROR: {description} failed with exit code {result.returncode}')
        return False

    print(f'\n✓ {description} completed successfully')
    return True


def train_base_method(track, tgt_lan, seed):
    """Train baseline (Dice coefficient) method."""
    if result_exists(track, tgt_lan, 'base', seed):
        print(f'\n[1/5] Baseline Method (Dice Coefficient) — skipping (results found)')
        return True
    cmd = f'python main.py --track {track} --tgt_lan {tgt_lan} --method base --seed {seed}'
    return run_command(cmd, f'[1/5] Baseline Method (Dice Coefficient)')


def train_mpnet(track, tgt_lan, seed):
    """Fine-tune MPNet with contrastive loss."""
    if ckpt_exists(track, tgt_lan, 'sbert', 'sentence-transformers/all-mpnet-base-v2', seed):
        print(f'\n[2/5] Fine-tune MPNet (Sentence Transformers) — skipping (checkpoint found)')
        return True
    cmd = f'python finetune.py --model_name mpnet --track {track} --tgt_lan {tgt_lan} --seed {seed}'
    return run_command(cmd, f'[2/5] Fine-tune MPNet (Sentence Transformers)')


def train_t5(track, tgt_lan, seed):
    """Fine-tune T5 with regression head."""
    if ckpt_exists(track, tgt_lan, 't5', 't5-base', seed):
        print(f'\n[3/5] Fine-tune T5 (Regression) — skipping (checkpoint found)')
        return True
    cmd = f'python finetune.py --model_name t5 --track {track} --tgt_lan {tgt_lan} --seed {seed}'
    return run_command(cmd, f'[3/5] Fine-tune T5 (Regression)')


def train_gpt2(track, tgt_lan, seed):
    """Fine-tune GPT-2 with regression head."""
    if ckpt_exists(track, tgt_lan, 'gpt2', 'gpt2', seed):
        print(f'\n[4/5] Fine-tune GPT-2 (Regression) — skipping (checkpoint found)')
        return True
    cmd = f'python finetune.py --model_name gpt2 --track {track} --tgt_lan {tgt_lan} --seed {seed}'
    return run_command(cmd, f'[4/5] Fine-tune GPT-2 (Regression)')


def train_roberta(track, tgt_lan, seed):
    """Fine-tune RoBERTa with regression head."""
    if ckpt_exists(track, tgt_lan, 'roberta', 'roberta-base', seed):
        print(f'\n[5/5] Fine-tune RoBERTa (Regression) — skipping (checkpoint found)')
        return True
    cmd = f'python finetune.py --model_name roberta --track {track} --tgt_lan {tgt_lan} --seed {seed}'
    return run_command(cmd, f'[5/5] Fine-tune RoBERTa (Regression)')


def generate_predictions(track, tgt_lan, seed):
    """Generate predictions for all trained methods."""
    methods = ['base', 'sbert', 't5', 'gpt2', 'roberta']
    print(f'\n{"="*70}')
    print('Generating predictions for all methods')
    print('='*70)

    for method in methods:
        cmd = f'python main.py --track {track} --tgt_lan {tgt_lan} --method {method} --seed {seed}'
        print(f'\nGenerating predictions: {method}...')
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'  WARNING: Prediction generation for {method} failed')
            print(result.stderr[-1000:] if result.stderr else '')
        else:
            print(f'  ✓ {method} predictions saved')

    print('\n✓ Prediction generation completed')


def train_ensemble(track, tgt_lan, seed):
    """Train XGBoost ensemble combining all method predictions."""
    methods = 'base,sbert,t5,gpt2,roberta'
    cmd = f'python ensemble.py --track {track} --tgt_lan {tgt_lan} --seed {seed} --methods {methods}'
    return run_command(cmd, f'[6/6] XGBoost Ensemble (XGB-4Ms)')


def evaluate_ensemble(config):
    """Evaluate ensemble predictions against ground truth."""
    ensemble_path = os.path.join(
        config.RESOURCE_PATH, 'results', config.track, config.tgt_lan, 'ensemble', str(config.seed),
        f'pred_{config.tgt_lan}_{config.track}.csv'
    )

    if not os.path.exists(ensemble_path):
        print(f'\nWARNING: Ensemble predictions not found at {ensemble_path}')
        return None

    ensemble_df = pd.read_csv(ensemble_path)
    ground_truth = pd.read_csv(config.DEV_LABEL_CSV)

    predictions = ensemble_df['Pred_Score'].values
    labels = ground_truth['Score'].values

    spearman = eva.get_spearman_cor(labels, predictions)
    pearson = eva.get_pearson_cor(labels, predictions)

    return {
        'spearman': spearman,
        'pearson': pearson,
        'predictions': predictions,
        'labels': labels
    }


def reproduce_paper(track, tgt_lan, seed, skip_methods=False):
    """
    Main reproduction workflow.

    Args:
        track: Task track (a, b, c, d, or sts)
        tgt_lan: Target language
        seed: Random seed
        skip_methods: If True, skip method training and only run ensemble
    """
    print('\n' + '='*70)
    print('SemEval-2024 Task 1: Semantic Textual Relatedness')
    print('Automated Paper Reproduction Workflow')
    print('='*70)

    print(f'\nConfiguration:')
    print(f'  Track: {track}')
    print(f'  Language: {tgt_lan}')
    print(f'  Seed: {seed}')

    config = Config()
    config.update_config(track=track, tgt_lan=tgt_lan, seed=seed)

    # Train all methods
    if not skip_methods:
        print(f'\n{"="*70}')
        print('PHASE 1: Train Individual Methods')
        print('='*70)

        success = True
        success = train_base_method(track, tgt_lan, seed) and success
        success = train_mpnet(track, tgt_lan, seed) and success
        success = train_t5(track, tgt_lan, seed) and success
        success = train_gpt2(track, tgt_lan, seed) and success
        success = train_roberta(track, tgt_lan, seed) and success

        if not success:
            print('\nWARNING: Some method training failed. Continuing with available methods...')

        # Generate predictions for all methods
        print(f'\n{"="*70}')
        print('PHASE 2: Generate Predictions')
        print('='*70)
        generate_predictions(track, tgt_lan, seed)

    # Train ensemble
    print(f'\n{"="*70}')
    print('PHASE 3: Train Ensemble')
    print('='*70)
    train_ensemble(track, tgt_lan, seed)

    # Evaluate
    print(f'\n{"="*70}')
    print('PHASE 4: Evaluation')
    print('='*70)

    results = evaluate_ensemble(config)
    if results:
        print(f'\nEnsemble Performance on Dev Set:')
        print(f'  Spearman Correlation: {results["spearman"]:.4f}')
        print(f'  Pearson Correlation: {results["pearson"]:.4f}')
        print(f'  Num Predictions: {len(results["predictions"])}')

    print(f'\n{"="*70}')
    print('Reproduction Completed Successfully!')
    print('='*70)

    print(f'\nFinal Results Location:')
    ensemble_results = os.path.join(
        config.RESOURCE_PATH, 'results', config.track, config.tgt_lan, 'ensemble', str(config.seed)
    )
    print(f'  {ensemble_results}/')

    print(f'\nNext Steps:')
    print(f'  1. Review individual method results: res/results/{track}/{tgt_lan}/')
    print(f'  2. Ensemble predictions: {ensemble_results}/')
    print(f'  3. Submit to competition or conduct further analysis')


if __name__ == '__main__':
    args = parse_args()

    try:
        reproduce_paper(args.track, args.tgt_lan, args.seed, skip_methods=args.skip_methods)
    except KeyboardInterrupt:
        print('\n\nReproduction interrupted by user.')
        sys.exit(1)
    except Exception as e:
        print(f'\n\nERROR: {e}')
        sys.exit(1)
