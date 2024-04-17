#!/bin/bash

sudo apt-get update
sudo apt-get upgrade
sudo apt install -y python3 python3-pip
sudo apt install -y fluidsynth
sudo apt install -y python3-tk
pip install -r requirements.txt
