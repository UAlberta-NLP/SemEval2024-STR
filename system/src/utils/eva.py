#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# public
import numpy as np
from scipy import stats


def get_spearman_cor(labels: list, preds: list) -> float:
    """
    Calculate Spearman rank correlation coefficient.

    Args:
        labels: Ground truth relatedness scores (list of floats, 0-1)
        preds: Predicted relatedness scores (list of floats, 0-1)

    Returns:
        Spearman correlation coefficient (float, -1 to 1)

    Raises:
        ValueError: If labels and preds have different lengths
    """
    if len(labels) != len(preds):
        raise ValueError("The length of labels and preds must be the same.")
    return stats.spearmanr(np.array(labels), np.array(preds))[0]


def get_pearson_cor(labels: list, preds: list) -> float:
    """
    Calculate Pearson correlation coefficient.

    Args:
        labels: Ground truth relatedness scores (list of floats, 0-1)
        preds: Predicted relatedness scores (list of floats, 0-1)

    Returns:
        Pearson correlation coefficient (float, -1 to 1)

    Raises:
        ValueError: If labels and preds have different lengths
    """
    if len(labels) != len(preds):
        raise ValueError("The length of labels and preds must be the same.")
    return stats.pearsonr(np.array(labels), np.array(preds))[0]