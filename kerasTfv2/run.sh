#!/bin/bash
set -e
CWD=`realpath $(dirname $0)`

build.sh
source ${CWD}/venv/bin/activate
python kerasAppQat.py
