#!/bin/sh

# FileName: install.sh
# Author: Jonathan Beaulieu
# Purpose: To update the system and to install missing/additional tools for the project.


sudo apt-get update
sudo apt-get install python3-pip

sudo -H pip3 install -r requirements.txt

# Grab glove twitter embeddings.
rm -rf data/glove.twitter
mkdir data/glove.twitter
cd data/glove.twitter
wget http://d.umn.edu/~beau0307/glove.twitter.27B.100d.txt
