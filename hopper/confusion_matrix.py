# Author: Jonathan Beaulieu
# Filename: confusion_matrix.py

# Purpose: To display an confusion matrix in the Output results. 
#Matrix is generated as a part of the classification output of predicted emojis against the gold emojis. 
from __future__ import division
from math import log10, ceil


def get_digits(num):
    # Gets the number of digits in a number
    return len(str(num))


class ConfusionMatrix(object):
    def __init__(self, dim):
        self.matrix = []
        for i in range(dim):
            self.matrix += [[0] * dim]

    def add(self, gold, out):
        self.matrix[gold][out] += 1

    def __str__(self):
        max_num = abs(max(map(max, self.matrix)))
        max_sum = max(map(sum, self.matrix))
        max_digits = get_digits(max_num)
        max_sum_digits = get_digits(max_sum)
        return "\n".join([str(i).rjust(2) + ": " + " ".join(map(lambda x: str(x).rjust(max_digits), r)) + "   ∑ = " + str(sum(r)).rjust(max_sum_digits) for i, r in enumerate(self.matrix)])
