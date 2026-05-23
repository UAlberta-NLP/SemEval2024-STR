"""
Abstract Meaning Representation (AMR) method for Semantic Textual Relatedness.

Parses sentences to Abstract Meaning Representations (AMR) and computes semantic similarity
using F-score matching between AMR graphs.

**Status**: Exploratory method (NOT in paper's official system). Requires external SPRING API
and the smatch module for AMR matching.

**Dependencies**:
1. SPRING API for AMR parsing: https://nlp.uniroma1.it/spring/api/text-to-amr
2. smatch module for AMR F-score matching:
   - Repository: https://github.com/mdtux89/amr-evaluation
   - Paper: "Smatch: an Evaluation Metric for Semantic Feature Structures" (Cai & Knight, 2013)
     https://aclanthology.org/P13-2131/
   - Installation: Clone the repo and copy smatch.py to system/src/smatch/

Usage:
  from amr import AMR
  model = AMR(config)
  scores, data = model.predict(sentences1, sentences2)

Note: This implementation is provided as a reference for the exploration done during
development. To use it fully, obtain the smatch module from the repository above.

References:
- AMR Parsing: https://github.com/amr-parsing
- SPRING API: https://nlp.uniroma1.it/spring/
- Smatch Evaluation: https://github.com/mdtux89/amr-evaluation
"""

import requests
from tqdm import tqdm
import pandas as pd
import os


def get_amr_match(amr1, amr2):
    """Stub AMR matching function. Replace with actual smatch implementation."""
    # TODO: Implement proper AMR F-score matching using smatch module
    # See: https://github.com/amr-parsing/smatch
    return (0.0, 0.0, 0.0)


def compute_f(precision, recall, named_entites_f=None):
    """Compute F-score from precision and recall."""
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f_score


class AMR(object):
    """
    Adapted from: https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024/blob/main/STR_Baseline.ipynb

    Uses SPRING API (https://nlp.uniroma1.it/spring/api/) for AMR parsing.
    F-score computation requires the smatch module (not included).
    """
    def __init__(self, config):
        super(AMR, self).__init__()
        self.config = config

    def smatch_score(self, s1, s2):
        amr1 = self.__get_amr(s1)
        amr2 = self.__get_amr(s2)
        scores = get_amr_match(amr1, amr2)
        p, r, f = compute_f(scores[0], scores[1], scores[2])
        return f, amr1, amr2

    def predict(self, s1s, s2s):
        scores = []
        data = []
        for s1, s2 in zip(tqdm(s1s), s2s):
            score, amr1, amr2 = self.smatch_score(s1, s2)
            scores.append(score)
            data.append({"s1": s1, "amr1": amr1, "s2": s2, "amr2": amr2, "score": score})
        return scores, data

    def __get_amr(self, s):
        params = {"sentence": s}
        r = requests.get("https://nlp.uniroma1.it/spring/api/text-to-amr", params=params)
        r = r.json()
        return r['penman']
