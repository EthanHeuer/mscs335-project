# MIDI Music Generation

## Description

This is a final project for MSCS 335 - Machine Learning at UW-Stout. The goal of this project is to generate music using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM). The model is trained on a dataset of MIDI files and generates new music in the form of a MIDI file.

**Authors:** Ethan Heuer & Andrew Jevne

## Installation

The following instructions are for setting up the project for VS Code in WSL.

### 1. Install Ubuntu for WSL:

1. Install WSL: [Guide](https://learn.microsoft.com/en-us/windows/wsl/install#install-wsl-command)
	```
	wsl --install
	```
2. Install Ubuntu via the Microsoft Store.
3. Update and upgrade Ubuntu:
	```
	sudo apt-get update && sudo apt-get upgrade
	```
4. Update Python and install pip:
    ```
    sudo apt install python3 python3-pip
    ```

### 2. Setup Workspace

1. Clone the repository into Ubuntu.
2. Open the project folder in VS Code:
   1. `cd` into the project folder.
   2. Run `code .` to open the project in VS Code.

### 3. Install Other Components

1. Install [FluidSynth](https://www.fluidsynth.org/):
   ```
   sudo apt install -y fluidsynth
   ```
2. Install other Python dependencies:
   ```
   pip install -r requirements.txt
   ```

<!--
### Setup virtual workspace

[learn.microsoft.com/en-us/windows/python/web-frameworks#install-python-pip-and-venv](https://learn.microsoft.com/en-us/windows/python/web-frameworks#install-python-pip-and-venv)

1. Update Python, install pip and venv:
	```
	sudo apt install python3 python3-pip python3-venv
	```
2. Create a virtual environment:
	```
	python3 -m venv .venv
	```
3. Activate the virtual environment:
	```
	source .venv/bin/activate
	```
-->

## Usage

*Work in progress...*
