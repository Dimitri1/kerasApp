#!/bin/bash

CWD=`realpath $(dirname $0)/..`

BAZEL_VERSION="0.15.2"
TENSORFLOW_VERSION="v1.12.3"

echo "solve tensorflow deps"
sudo apt install python-dev
sudo apt install python3-pip
sudo apt-get install openjdk-8-jdk
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3

pip3 install -U --user keras_applications==1.0.8 --no-deps
pip3 install -U --user keras_preprocessing==1.1.0 --no-deps
pip3 install -U --user numpy==1.16.0
pip3 install -U --user tensorflow==1.12.0
pip3 install -U --user pip six wheel setuptools mock matplotlib

TENSORFLOW_SOURCE_PARENT_DIR=$CWD/external
mkdir -p $TENSORFLOW_SOURCE_PARENT_DIR

TENSORFLOW_SOURCE_DIR=$TENSORFLOW_SOURCE_PARENT_DIR/tensorflow

# install bazel, the tensorflow build system
curl -L https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --output bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user
export PATH="$PATH:$HOME/bin"

# download tensorflow in vsora/external
cd ${TENSORFLOW_SOURCE_PARENT_DIR}
rm -rf tensorflow
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout $TENSORFLOW_VERSION -b $TENSORFLOW_VERSION
git apply ../tensorflow.patch

# configuration
./configure

# build various tools
bazel build tensorflow/examples/label_image:label_image
bazel build tensorflow/tools/graph_transforms:transform_graph
bazel build tensorflow/python/tools:freeze_graph
bazel build tensorflow/tools/graph_transforms:summarize_graph
