#!/bin/bash
set -e
CWD=`realpath $(dirname $0)`

build.sh
source ${CWD}/venv/bin/activate
python ${CWD}/TensorFlow-Examples/tensorflow_v1/examples/3_NeuralNetworks/convolutional_network_raw.py
