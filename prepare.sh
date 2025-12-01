#!/bin/bash

pip install torch torchvision torchaudio torchtext --force-reinstall

pip install -r requirements.txt
pip install -v -e .

echo "accelerate config done"
