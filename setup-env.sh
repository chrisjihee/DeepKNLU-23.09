#!/bin/bash
# for running program
conda create -n DeepKNLU-23.09 python=3.10 -y
conda activate DeepKNLU-23.09
pip install -r requirements.txt
pip list

# for editing library
rm -rf chrisbase* chrislab*
pip download --no-binary :all: --no-deps chrisbase==0.4.5; tar zxf chrisbase-*.tar.gz; rm chrisbase-*.tar.gz;
pip download --no-binary :all: --no-deps chrislab==0.6.0; tar zxf chrislab-*.tar.gz; rm chrislab-*.tar.gz;
pip install --editable chrisbase-*
pip install --editable chrislab-*

# for developing library
rm -rf chrisbase* chrislab*
git clone git@github.com:chrisjihee/chrisbase.git
git clone git@github.com:chrisjihee/chrislab.git
pip install --editable chrisbase
pip install --editable chrislab
