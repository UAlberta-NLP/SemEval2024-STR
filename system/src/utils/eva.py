#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# public
import numpy as np
from scipy import stats


def get_spearman_cor(labels: list, preds: list) -> float:
    # check if labels and preds have the same length
    if len(labels) != len(preds):
        raise ValueError("The length of labels and preds must be the same.")
    # calculate and return spearman correlation
    return stats.spearmanr(np.array(labels), np.array(preds))[0]