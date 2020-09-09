#!/bin/bash
set -e
CWD=`realpath $(dirname $0)`
virtualenv  --python=python3.6 ${CWD}/venv-qat-fcustom

source ${CWD}/venv-qat-fcustom/bin/activate
pip install -q tf-nightly-gpu
pip install -q tensorflow-model-optimization

nvidia-smi

pip install matplotlib
pip install tensorflow_datasets
pip install tensorflow=='2.3'
