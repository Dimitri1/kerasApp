$1/bin/pip uninstall -y tensorflow tensorflow-gpu
$1/bin/pip install -q tf-nightly-gpu
$1/bin/pip install -q tensorflow-model-optimization

show tf-nightly-gpu
show tensorflow-model-optimization
nvidia-smi

$1/bin/pip install matplotlib
$1/bin/pip install tensorflow_datasets