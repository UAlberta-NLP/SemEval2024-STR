#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import re


class Model(object):
    """
    adapted from https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024/blob/main/STR_Baseline.ipynb
    """
    def __init__(self):
        super(Model, self).__init__()

    def dice_score(self, s1, s2):
        s1 = s1.lower()
        s1_split = re.findall(r"\w+|[^\w\s]", s1, re.UNICODE)
        s2 = s2.lower()
        s2_split = re.findall(r"\w+|[^\w\s]", s2, re.UNICODE)
        dice_coef = len(set(s1_split).intersection(set(s2_split))) / (len(set(s1_split)) + len(set(s2_split)))
        return round(dice_coef, 2)

    def predict(self, s1s, s2s):
        return [self.dice_score(s1, s2) for s1, s2 in zip(s1s, s2s)]