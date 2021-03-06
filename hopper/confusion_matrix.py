# Author: Jonathan Beaulieu
# Filename: confusion_matrix.py

# Purpose: To display an confusion matrix in the Output results.
# Matrix is generated as a part of the classification output of predicted emojis against the gold emojis.
from __future__ import division

import json

from .scorer import Scorer


def get_digits(num):
    # Gets the number of digits in a number
    return len(str(num))


class ConfusionMatrix(Scorer):
    def __init__(self, dim):
        super().__init__()
        self.matrix = []
        for i in range(dim):
            self.matrix += [[0] * dim]

    def add(self, gold, out):
        super().add(gold, out)
        self.matrix[gold][out] += 1

    def __str__(self):
        max_num = abs(max(map(max, self.matrix)))
        max_sum = max(map(sum, self.matrix))
        max_digits = get_digits(max_num)
        max_sum_digits = get_digits(max_sum)
        return "\n".join([str(i).rjust(2) + ": " + " ".join(
            map(lambda x: str(x).rjust(max_digits), r)) + "   ∑ = " + str(sum(r)).rjust(max_sum_digits) for i, r in
                          enumerate(self.matrix)])


class ConfusionMatrixWithExamples(ConfusionMatrix):
    def __init__(self, dim):
        super().__init__(dim)
        self.example_matrix = [[[] for _ in range(dim)] for _ in range(dim)]

    def add(self, gold, out, text, tokens):
        super().add(gold, out)
        self.example_matrix[gold][out] += [(text, tokens)]

    def dump_json(self, fn):
        json.dump(self.example_matrix, open(fn, "w"))
