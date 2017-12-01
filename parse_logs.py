import os
import re
import glob

import shutil


flatten = lambda l: [item for sublist in l for item in sublist]


def get_row_str(l):
    return "{}: ".format(str(max(l)).ljust(6)) + " ".join(map(lambda x: str(x).ljust(7), l))


def get_score(l):
    return str(max(l))


def main():
    sys_dirs = []
    for x in os.listdir("output"):
        p = os.path.join("output", x)
        if os.path.isdir(p):
            sys_dirs += [p]
    # os.path.abspath("output") + "/"+x for x in  if ]
    # sys_dirs = glob.glob("output/char_*")
    # print(g)
    # return
    # sys_dirs = os.listdir("logs")
    logs = []
    for d in sys_dirs:
        g = glob.glob(os.path.join(d, "*.log"))
        if len(g) > 1 and os.path.join(d, "log.log") in g:
            logs += [os.path.join(d, "log.log")]
        else:
            assert len(g) == 1, "Too many log files for '{}': {}".format(d, g)
            logs += [g[0]]

    sys_results_macro = {}
    sys_results_micro = {}
    for log in logs:
        txt = open(log, "r").read()
        res = re.findall(r"----- Iteration ([0-9]{1,3}) results for fold ([0-9]{1,3}) -----\nMacro F1: ([0-9\.]{3,7})\n-----\nMicro F1: ([0-9\.]{3,7})", txt, re.MULTILINE)
        obj_mac = {}
        obj_mic = {}
        for r in res:
            iteration, fold, mac_score, mic_score = r
            iteration, fold, mac_score, mic_score = int(iteration), int(fold), float(mac_score), float(mic_score)
            if fold not in obj_mac:
                obj_mac[fold] = {}
                obj_mic[fold] = {}
            obj_mac[fold][iteration] = mac_score
            obj_mic[fold][iteration] = mic_score
        sys_results_macro[os.path.split(os.path.split(log)[0])[1]] = obj_mac
        sys_results_micro[os.path.split(os.path.split(log)[0])[1]] = obj_mic

        if len(res) == 0:
            # The log isn't a nn log.
            res = re.findall(r"----- Results -----\nMacro F1: ([0-9\.]{3,7})\n-----\nMicro F1: ([0-9\.]{3,7})", txt, re.MULTILINE)
            if len(res) == 0:
                res = re.findall(r"----- Results for fold [0-9]{1,3} -----\nMacro F1: ([0-9\.]{3,7})\n-----\nMicro F1: ([0-9\.]{3,7})", txt, re.MULTILINE)
            obj_mac = {}
            obj_mic = {}
            for i, (mac_score, mic_score) in enumerate(res):
                obj_mac[i] = {0: float(mac_score)}
                obj_mic[i] = {0: float(mic_score)}
            sys_results_macro[os.path.split(os.path.split(log)[0])[1]] = obj_mac
            sys_results_micro[os.path.split(os.path.split(log)[0])[1]] = obj_mic

    # Turn the dictionary into arrays
    max_fold = max(flatten([r.keys() for r in sys_results_macro.values()])) + 1
    max_iter = max(flatten([l.keys() for r in sys_results_macro.values() for l in r.values()])) + 1

    # print(max_fold, max_iter)
    # print(sys_results)

    results_array_macro = {}
    for name, res in sys_results_macro.items():
        res_array = []
        for fold in res.values():
            fold_res = []
            for score in fold.values():
                fold_res += [score]
                # results_array[j][iteration] = score
            res_array += [fold_res]
        results_array_macro[name] = res_array

    results_array_micro = {}
    for name, res in sys_results_micro.items():
        res_array = []
        for fold in res.values():
            fold_res = []
            for score in fold.values():
                fold_res += [score]
                # results_array[j][iteration] = score
            res_array += [fold_res]
        results_array_micro[name] = res_array

    # print(results_array)
    print("Key\nname_of_configuration\nMax Macro-F1: Macro-F1 for each epoch starting with the first and going to the last.")
    for name, v in sorted(results_array_macro.items()):
        print(name, "\n" + "\n".join(map(get_row_str, v)))

    with open("results_macro.csv", "w") as fp:
        print("Model Id," + ",".join("Fold {}".format(i+1) for i in range(max_fold)), file=fp)
        for name, v in sorted(results_array_macro.items()):
            print(name + ",", end="", file=fp)
            print(",".join(map(get_score, v)), file=fp)

    with open("results_micro.csv", "w") as fp:
        print("Model Id," + ",".join("Fold {}".format(i+1) for i in range(max_fold)), file=fp)
        for name, v in sorted(results_array_micro.items()):
            print(name + ",", end="", file=fp)
            print(",".join(map(get_score, v)), file=fp)


if __name__ == "__main__":
    main()
