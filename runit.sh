#!/bin/sh
# Author: Jonathan Beaulieu

# Use this command to train any of the configurations in the config directory.
# python3 run.py configs/bow_en.json -v

# run 'python3 run.py -h' to print the following help text.
# usage: run.py [-h] [-f FOLD] [-e] [-v] [-r] config
#
# positional arguments:
#   config
#
# optional arguments:
#   -h, --help            show this help message and exit
#   -f FOLD, --fold FOLD (Runs a single fold instead of all folds.)
#   -e, --examples (This saves a confusion matrix with examples so you can explorer how the tweets are being classified, using the the matrix_explorer.py program.)
#   -v, --verbose (This makes enables all logging statements to log not just the essential ones. Good for being able see what is/was going on.)
#   -r, --recover (This is an experimental feature)

# This command gets the results from the trained models. (Don't comment out even if you uncomment the above command to train a model, otherwise the results will not update.)
python3 parse_output.py
