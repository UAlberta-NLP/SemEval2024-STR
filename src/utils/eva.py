#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import numpy as np
from scipy import stats


def get_spearman_cor(labels, preds):
    return stats.spearmanr(np.array(labels), np.array(preds))[0]