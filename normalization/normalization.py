# Author : Sai Cheedella

#Purpose: To normalise the given text data. For our project we are normalising english and spanish twitter data to train our models.

from __future__ import division
import os
from normalise.normalisation import normalise  # imports the normalisation module
import re

languages = ['us','es']
# to load the language specific train data
for language in languages:
   file = open(os.getcwd() + '/' + language + '_train.text', 'r')
   output_file = open(os.getcwd() + '/' + language + '_train_normal.text', 'w')
# string manipulation operations
   for line in file:
      line = re.sub(r'[\s]+\)', ')', re.sub(r'\([\s]+', '(', line)) # regex to remove a whitespace before and after a paranthesis
      line = re.sub(r'[\s]+\]', ']', re.sub(r'\[[\s]+', '[', line)) # regex to remove a whitespace before and after a square bracket
      line = re.sub(r'[\s]+\}', '}', re.sub(r'\{[\s]+', '{', line)) # regex to remove a whitespace before and after a flower bracket
      line = re.sub(r'(?i)\b((?:https?:\/\/|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', line)   # regex to remove the urls and twitter links
      line = re.sub(r'-', '', line) # regex to remove '-'
      line = re.sub(r'(.)\1{9,}', '', line) #regex to remove repetitive characters
      z = normalise(line, verbose=False, variety="Ame")  # calling the normalise function, verbose mode: prints out complete run of program, variety: to specify as american english
      print(z, file=output_file) # writes the data to an output file
