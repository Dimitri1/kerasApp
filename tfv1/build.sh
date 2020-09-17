#!/bin/bash
set -e
CWD=`realpath $(dirname $0)`
virtualenv  --python=python3.6 ${CWD}/venv

source ${CWD}/venv/bin/activate
pip install -q tf-nightly-gpu
pip install -q tensorflow-model-optimization

nvidia-smi

pip install matplotlib
pip install tensorflow_datasets
pip install tensorflow
pip install tensorflow-1.14.0-cp36-cp36m-linux_x86_64.whl
