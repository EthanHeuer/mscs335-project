# Midi Music Generation

## Description

This is a final project for MSCS 335 - Machine Learning at UW-Stout. The goal of this project is to generate music using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM). The model is trained on a dataset of MIDI files and generates new music in the form of a MIDI file.

**Authors:** Ethan Heuer & Andrew Jevne

## Getting Started

### Installation

FluidSynth is required to run the project. Installing FluidSynth on Windows is a ***pain***, so it is recommended to use a Linux environment. The following instructions are for setting up the project on Windows Subsystem for Linux (WSL).

1. Install Ubuntu for WSL:
   1. Install WSL: [learn.microsoft.com/en-us/windows/wsl/install](https://learn.microsoft.com/en-us/windows/wsl/install#install-wsl-command)
        ```
        wsl --install
        ```
   2. Install Ubuntu via the Microsoft Store.
   3. Update and upgrade Ubuntu:
        ```
        sudo apt-get update && sudo apt-get upgrade
        ```
   4. Clone the repository into Ubuntu.
2. Install pip:
   ```
   sudo apt install python3-pip
   ```
3. Install FluidSynth: [fluidsynth.org/download/](https://www.fluidsynth.org/download/)
   ```
   sudo apt install -y fluidsynth
   ```
4. Install tensorflow: [tensorflow.org/install/pip#windows-wsl2_1](https://www.tensorflow.org/install/pip#windows-wsl2_1)
   ```
   pip install tensorflow[and-cuda]
   ```
5. Install other python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

*Work in progress...*
