#!/bin/bash
conda create -n DeepKNLU-23.07 python=3.10 -y
conda activate DeepKNLU-23.07
pip install -r requirements.txt
pip list

rm -rf chrisbase chrislab
git clone git@github.com:chrisjihee/chrisbase.git
git clone git@github.com:chrisjihee/chrislab.git
pip install --editable chrisbase
pip install --editable chrislab

rm -f pretrained-com pretrained-pro
ln -s ../pretrained-com .
ln -s ../pretrained-pro .
