#!/bin/bash
# for running program
conda create -n DeepKNLU-23.09 python=3.10 -y
conda activate DeepKNLU-23.09
pip install -r requirements.txt
pip list

# for editing library
rm -rf chrisbase* chrislab*
pip download --no-binary :all: --no-deps chrisbase==0.4.6; tar zxf chrisbase-*.tar.gz; rm chrisbase-*.tar.gz;
pip download --no-binary :all: --no-deps chrislab==0.6.1; tar zxf chrislab-*.tar.gz; rm chrislab-*.tar.gz;
pip install --editable chrisbase-*
pip install --editable chrislab-*

# for developing library
rm -rf chrisbase* chrislab*
git clone https://github.com/chrisjihee/chrisbase.git
git clone https://github.com/chrisjihee/chrislab.git
pip install --editable chrisbase
pip install --editable chrislab

# for pretrained model
git lfs install
git clone https://github.com/KPFBERT/kpfbert pretrained/KPF-BERT
git clone https://huggingface.co/klue/bert-base pretrained/KLUE-BERT
git lfs uninstall

git clone guest@129.254.164.137:git/pretrained-com
