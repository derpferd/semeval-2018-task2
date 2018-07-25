import os
import shutil

import sys

from hopper.model_svm import LinearSVCModel, LinearSVC2Model


def get_p_bar(i, t):
    stars = round((i/t)*78)
    spaces = 78-stars
    return "\r"+"["+"*"*stars+" "*spaces+"]"


def test(model, in_fp, out_fp):
    print("\tReading input...")
    texts = in_fp.read().split("\n")
    print("\tRunning through model and writing output...")
    predictions = model.batch_predict(texts)
    i = 1
    print()
    for pred in predictions:
        out_fp.write(str(pred) + "\n")
        if i % 100 == 0:
            sys.stdout.write(get_p_bar(i, len(texts)))
        i += 1
    sys.stdout.write(get_p_bar(1, 1))


def main():
    print("Starting...")
    # test english
    final_dir = "final_test"
    in_dir = os.path.join(final_dir, "test_semeval2018task2_text")
    out_dir = os.path.join(final_dir, "test_output")

    print("Cleaning up...\n")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    print("Creating English Model...")
    model = LinearSVCModel()
    print("Loading Saved English Model...")
    model.load_model("final_test/lsvc_en_final")
    print("Testing English Model...")
    with open(os.path.join(in_dir, "us_test.text"), "r") as in_fp, open(os.path.join(out_dir, "english.output.txt"), "w") as out_fp:
        test(model, in_fp, out_fp)
    print()

    print("Creating Spanish Model...")
    model = LinearSVCModel()
    print("Loading Saved Spanish Model...")
    model.load_model("final_test/lsvc_2_es_final")
    print("Testing Spanish Model...")
    with open(os.path.join(in_dir, "es_test.text"), "r") as in_fp, open(os.path.join(out_dir, "spanish.output.txt"), "w") as out_fp:
        test(model, in_fp, out_fp)
    print()

    print("Done.")


if __name__ == "__main__":
    main()
