#!/bin/bash
conda create -n DeepKNLU-23.09 python=3.10 -y
conda activate DeepKNLU-23.09
pip install -r requirements.txt
pip list

# optional for editing library
rm -rf chrisbase chrislab
git clone git@github.com:chrisjihee/chrisbase.git
git clone git@github.com:chrisjihee/chrislab.git
pip install --editable chrisbase
pip install --editable chrislab
