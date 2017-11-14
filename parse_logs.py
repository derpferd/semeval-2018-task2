import os
import re
import glob

import shutil

flatten = lambda l: [item for sublist in l for item in sublist]


def get_row_str(l):
    return "{}: ".format(str(max(l)).ljust(6)) + " ".join(map(lambda x: str(x).ljust(7), l))
    # return "{},".format(str(max(l))) + ",".join(map(lambda x: str(x), l))


def main():
    sys_dirs = glob.glob("output/char_*")
    # print(g)
    # return
    # sys_dirs = os.listdir("logs")
    logs = []
    for d in sys_dirs:
        g = glob.glob(os.path.join(d, "*.log"))
        assert len(g) == 1
        logs += [g[0]]

    sys_results = {}
    for log in logs:
        txt = open(log, "r").read()
        res = re.findall(r"----- Iteration ([0-9]{1,3}) results for fold ([0-9]{1,3}) -----\nMacro F1: ([0-9\.]{3,7})", txt, re.MULTILINE)
        obj = {}
        for r in res:
            iteration, fold, score = r
            iteration, fold, score = int(iteration), int(fold), float(score)
            if fold not in obj:
                obj[fold] = {}
            obj[fold][iteration] = score
        sys_results[os.path.split(os.path.split(log)[0])[1]] = obj

    # Turn the dictionary into arrays
    max_fold = max(flatten([r.keys() for r in sys_results.values()])) + 1
    max_iter = max(flatten([l.keys() for r in sys_results.values() for l in r.values()])) + 1

    # print(max_fold, max_iter)
    # print(sys_results)

    # results_array = [[[None] * max_iter] * max_fold] * len(sys_results)
    results_array = {}

    for name, res in sys_results.items():
        res_array = []
        for fold in res.values():
            fold_res = []
            for score in fold.values():
                fold_res += [score]
                # results_array[j][iteration] = score
            res_array += [fold_res]
        results_array[name] = res_array

    # print(results_array)
    print("Key\nname_of_configuration\nMax Macro-F1: Macro-F1 for each epoch starting with the first and going to the last.")
    for name, v in sorted(results_array.items()):
        print(name, "\n" + "\n".join(map(get_row_str, v)))


if __name__ == "__main__":
    main()
