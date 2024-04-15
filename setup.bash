#!/bin/bash

sudo apt-get update
sudo apt-get upgrade
sudo apt install python3 python3-pip
sudo apt install -y fluidsynth
pip install -r requirements.txt
code .
