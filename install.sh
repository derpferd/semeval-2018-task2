#!/bin/sh

# FileName: install.sh
# Author: Jonathan Beaulieu
# Purpose: To update the system and to install missing/additional tools for the project.


sudo apt-get update
sudo apt-get install python3-pip

sudo -H pip3 install -r requirements.txt
