#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'

"""
Submission utilities for formatting and validating predictions.

Provides functions to load predictions from CSV files, format them according
to competition requirements, validate score ranges, and prepare submission files.
"""

import os
import zipfile
import pandas as pd
import numpy as np


def load_predictions(csv_path):
    """
    Load predictions from a CSV file.

    Args:
        csv_path: Path to CSV with columns [PairID, Pred_Score]

    Returns:
        pandas DataFrame with columns [PairID, Pred_Score]

    Raises:
        FileNotFoundError: If CSV file does not exist
        ValueError: If CSV is missing required columns
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prediction file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = ['PairID', 'Pred_Score']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV missing required columns: {missing_columns}")

    return df


def save_submission(predictions_df, output_path):
    """
    Save predictions to submission format CSV.

    Ensures the output has exactly two columns [PairID, Pred_Score] as required
    by the competition.

    Args:
        predictions_df: DataFrame with columns [PairID, Pred_Score]
        output_path: Path where to save the CSV file

    Returns:
        Path to saved CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_df = predictions_df[['PairID', 'Pred_Score']].copy()
    output_df.to_csv(output_path, index=False)

    return output_path


def round_predictions(scores, decimals=2):
    """
    Round prediction scores to specified decimal places and clip to valid range.

    Args:
        scores: Numeric array or list of predictions
        decimals: Number of decimal places to round to (default: 2)

    Returns:
        numpy array of rounded and clipped scores in range [0, 1]
    """
    scores = np.asarray(scores)
    rounded = np.round(scores, decimals)
    clipped = np.clip(rounded, 0.0, 1.0)
    return clipped


def validate_submissions(predictions_df):
    """
    Validate that predictions meet competition requirements.

    Checks:
    - All scores are in range [0, 1]
    - No NaN or infinite values
    - PairID column exists and is numeric
    - Pred_Score column exists

    Args:
        predictions_df: DataFrame to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    if 'PairID' not in predictions_df.columns:
        raise ValueError("Missing PairID column")

    if 'Pred_Score' not in predictions_df.columns:
        raise ValueError("Missing Pred_Score column")

    scores = predictions_df['Pred_Score'].values

    if np.any(np.isnan(scores)):
        raise ValueError("Predictions contain NaN values")

    if np.any(np.isinf(scores)):
        raise ValueError("Predictions contain infinite values")

    if np.any((scores < 0) | (scores > 1)):
        invalid_count = np.sum((scores < 0) | (scores > 1))
        raise ValueError(f"{invalid_count} predictions outside [0, 1] range")

    return True


def zip_submission(csv_path, zip_path):
    """
    Create a zip file containing the submission CSV.

    Args:
        csv_path: Path to the prediction CSV file
        zip_path: Path where to save the zip file

    Returns:
        Path to created zip file
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        arcname = os.path.basename(csv_path)
        zipf.write(csv_path, arcname=arcname)

    return zip_path


def load_and_validate(csv_path):
    """
    Load predictions and validate them in one step.

    Args:
        csv_path: Path to prediction CSV

    Returns:
        Validated DataFrame with predictions

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If validation fails
    """
    df = load_predictions(csv_path)
    validate_submissions(df)
    return df
