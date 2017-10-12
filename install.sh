#!/bin/sh
# Author: Jonathan Beaulieu

sudo apt-get update
sudo apt-get install python3-pip

sudo -H pip3 install -r requirements.txt
