#!/bin/bash
set -e
CWD=`realpath $(dirname $0)`
virtualenv  --python=python3.6 ${CWD}/venv-qat-fcustom
${CWD}/install_pip_packages.sh ${CWD}/venv-qat-fcustom
