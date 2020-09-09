import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def save(model, pb):
    frozen_func = convert_variables_to_constants_v2(
        tf.function(lambda x: model(x)).get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape,
                          model.inputs[0].dtype,
                          name="input")))
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name=pb,
                      as_text=False)
