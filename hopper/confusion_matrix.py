# Author: Jonathan Beaulieu
from __future__ import division
from math import log10, ceil


def get_digits(num):
    if num == 0:
        num = 1
    if (num / 10.0) == (num // 10):
        num += 1
    return int(ceil(log10(num)))


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
