from __future__ import division
from collections import defaultdict


def f1(precision,recall):
    return (2.0*precision*recall)/(precision+recall)


class Score(object):
    def __init__(self, macro_f1, micro_f1, p, r):
        self.macro_f1 = macro_f1
        self.micro_f1 = micro_f1
        self.percision = p
        self.recall = r

    def __str__(self):
        s = "Macro F1: {}\n-----\nMicro F1: {}\nPrecision: {}\nRecall: {}".format(*map(lambda x: round(x*100, 3), [self.macro_f1, self.micro_f1, self.percision, self.recall]))
        return s


class Scorer(object):
    def __init__(self):
        self.truth_dict=defaultdict(int)
        self.output_dict_correct=defaultdict(int)
        self.output_dict_attempted=defaultdict(int)

    def add(self, gold, out):
        self.truth_dict[gold] += 1
        if out == gold:
            self.output_dict_correct[out] += 1
        self.output_dict_attempted[out] += 1

    def get_score(self):
        """Returns the score so far."""
        precision_total = 0
        recall_total = 0
        num_emojis = len(self.truth_dict)
        attempted_total = 0
        correct_total = 0
        gold_occurrences_total = 0
        f1_total = 0
        for emoji_code in self.truth_dict:
            gold_occurrences = self.truth_dict[emoji_code]
            attempted = self.output_dict_attempted.get(emoji_code, 0)
            correct = self.output_dict_correct.get(emoji_code, 0)
            if attempted != 0:
                precision = correct / attempted
                recall = correct / gold_occurrences
                if precision != 0 or recall != 0:
                    f1_total += f1(precision, recall)
            attempted_total += attempted
            correct_total += correct
            gold_occurrences_total += gold_occurrences
        macrof1 = f1_total/num_emojis
        precision_total_micro = correct_total/attempted_total
        recall_total_micro = correct_total/gold_occurrences_total
        if precision_total_micro != 0 or recall_total_micro != 0:
            microf1 = f1(precision_total_micro,recall_total_micro)
        else:
            microf1 = None
        return Score(macrof1, microf1, precision_total_micro, recall_total_micro)


