#!/bin/bash

#install mamba into base conda env
conda install mamba 

# Create the Conda environment at the specified directory
mamba create --prefix ./env_LocalDiscoPub python=3.12

# Activate the environment using its path
conda activate ./env_LocalDiscoPub

# Install packages from the requirements.txt file
pip install -r requirements.txt