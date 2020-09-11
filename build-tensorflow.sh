#!/bin/bash

CWD=`realpath $(dirname $0)`

BAZEL_VERSION="0.25.2"

echo "solve tensorflow deps"
#sudo apt install python-dev
#sudo apt install python3-pip
#sudo apt-get install openjdk-8-jdk
#sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3
#pip3 install -U --user keras_applications==1.0.8 --no-deps
#pip3 install -U --user keras_preprocessing==1.1.0 --no-deps
#pip3 install -U --user numpy==1.16.0
#pip3 install -U --user tensorflow==1.12.0
#pip3 install -U --user pip six wheel setuptools mock matplotlib

TENSORFLOW_SOURCE_PARENT_DIR=$CWD/tensorflow
mkdir -p $TENSORFLOW_SOURCE_PARENT_DIR

TENSORFLOW_SOURCE_DIR=$TENSORFLOW_SOURCE_PARENT_DIR/tensorflow

# install bazel, the tensorflow build system
curl -L https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --output bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user

export PATH="$PATH:$HOME/bin"

cd ${TENSORFLOW_SOURCE_PARENT_DIR}
./configure
bazel build //tensorflow/tools/pip_package:build_pip_package
