from __future__ import print_function
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

tensorflow.compat.v1.disable_eager_execution()

batch_size = 128
num_classes = 10
epochs = 5

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


import tensorflow_datasets as tfds

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

splits = ("train[:80%]", "train[:10%]", "train[:10%]")
IMG_SIZE = 224
SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 32
(raw_train, raw_validation, raw_test), info = tfds.load(name="tf_flowers",
                                                        with_info=True,
                                                        split=list(splits),
                                                        as_supervised=True)
#def setup_mobilenet_v2_model():
#  import tensorflow as tf
#  base_model = MobileNetV2(include_top=False,
#                           weights='imagenet',
#                           pooling='avg',
#                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
#  x = base_model.output
#  x = tf.keras.layers.Dense(info.features['label'].num_classes, activation="softmax")(x)
#  model_functional = tf.keras.Model(inputs=base_model.input, outputs=x)
#
#  return model_functional
#
#mode = setup_mobilenet_v2_model()
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(x_test, y_test))
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])





import os
os.makedirs('./model', exist_ok=True)
#model.save('./model/keras_model.h5')

#K.set_learning_phase(0)


print(tensorflow.__version__)
sess = tensorflow.compat.v1.keras.backend.get_session()
graph_def = sess.graph.as_graph_def()

# graph_def
#show_graph(graph_def)

model.summary()
print(model.outputs)
print(model.inputs)



def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
exit(0)
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tensorflow.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tensorflow.compat.v1.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        print('outout ', output_names)
        print('graph ', input_graph_def)
        print('freeze_var ', freeze_var_names)

        #gd = tensorflow.compat.v1.graph_util.convert_variables_to_constants(
        #                        sess, input_graph_def, output_names)

        #frozen_graph = tensorflow.compat.v1.graph_util.extract_sub_graph(gd, output_names)
        frozen_graph = tensorflow.compat.v1.graph_util.extract_sub_graph(input_graph_def, output_names)
        return frozen_graph



tensorflow.saved_model.save(model, 'model')
model.save('./model/saved_model.h5')

exit(0)
sess=tensorflow.compat.v1.InteractiveSession()
frozen_graph = freeze_session(sess,
                              output_names=[out.op.name for out in model.outputs])




tensorflow.io.write_graph(frozen_graph, "model", "tensorflow.model.pb", as_text=False)
exit(0)

import tensorflow as tf
import os
import sys
from tensorflow.python.platensorflow.rm import gfile

sess=tensorflow.InteractiveSession()


f = gfile.FastGFile("./model/tensorflow.model.pb", 'rb')
graph_def = tensorflow.GraphDef()
# Parses a serialized binary message into the current message.
graph_def.ParseFromString(f.read())
f.close()




sess.graph.as_default()
# Import a serialized TensorFlow `GraphDef` protocol buffer
# and place into the current default `Graph`.
tensorflow.import_graph_def(graph_def)


softmax_tensor = sess.graph.get_tensor_by_name('import/dense_2/Softmax:0')


