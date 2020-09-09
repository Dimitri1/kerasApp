# -*- coding: utf-8 -*-

import sys
import os
import datetime
import time

import numpy as np

if sys.version_info.major >= 3:
    import pathlib
else:
    import pathlib2 as pathlib

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow_model_optimization as tfmo

"""# Prepare the dataset"""

splits = ("train[:80%]", "train[:10%]", "train[:10%]")

# For tf_flowers
(raw_train, raw_validation, raw_test), info = tfds.load(name="tf_flowers",
                                                        with_info=True,
                                                        split=list(splits),
                                                        as_supervised=True)

# For cats vs dogs
# (raw_train, raw_validation, raw_test), info = tfds.load(name="cats_vs_dogs",
#                                                         with_info=True,
#                                                         split=list(splits),
#                                                         as_supervised=True)

"""Show datasets info"""
exit(0)
info

total_num_examples = info.splits['train'].num_examples

temp_num = total_num_examples - (total_num_examples % 100)
num_train = int(temp_num * 0.8 + (70 - 14))
num_val =  int(temp_num * 0.1 + 7)
num_test = int(temp_num * 0.1 + 7)

print('num train:', num_train)
print('num val  :', num_val)
print('num test :', num_test)

IMG_SIZE = 224
SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 32

"""## Data augumentation"""

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = image / 255.0
    # image = (image/127.5) - 1

    # Resize the image if required
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

"""## Input format"""

def augment_data(image, label):
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_contrast(image, lower=0.1, upper=0.6)

  return image, label

train = train.map(augment_data)

train = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation = validation.batch(BATCH_SIZE)
test = test.batch(1)

# (Optional) prefetch will enable the input pipeline to asynchronously fetch batches while
# your model is training.
train = train.prefetch(tf.data.experimental.AUTOTUNE)

print(train)
print(validation)
print(test)

"""## Display train image"""

# Get the function which converts label indices to string
get_label_name = info.features['label'].int2str

plt.figure(figsize=(12,12)) 

for batch in train.take(1):
  for i in range(9):
    image, label = batch[0][i], batch[1][i]
    plt.subplot(3, 3, i+1)
    plt.imshow(image.numpy())
    plt.title(get_label_name(label.numpy()))
    plt.grid(False)

"""# Train model

## Build model
"""

def setup_cnn_model():
  # extract image features by convolution and max pooling layers
  inputs = tf.keras.Input(shape = (IMG_SIZE, IMG_SIZE, 3))
  x = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(inputs)
  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
  # classify the class by fully-connected layers
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(512, activation="relu")(x)
  x = tf.keras.layers.Dense(info.features['label'].num_classes)(x)
  x = tf.keras.layers.Activation("softmax")(x)
  model_functional = tf.keras.Model(inputs=inputs, outputs=x)

  return model_functional

def setup_mobilenet_v2_model():
  base_model = MobileNetV2(include_top=False,
                           weights='imagenet',
                           pooling='avg',
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
  x = base_model.output
  x = tf.keras.layers.Dense(info.features['label'].num_classes, activation="softmax")(x)
  model_functional = tf.keras.Model(inputs=base_model.input, outputs=x)

  return model_functional

model = setup_mobilenet_v2_model()

model.trainable = True

model.summary()

base_learning_rate = 0.0001
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss = 'sparse_categorical_crossentropy',
              metrics = ["accuracy"])

"""## Training (Not QAT)"""

steps_per_epoch = round(num_train) // BATCH_SIZE
validation_steps = round(num_val) // BATCH_SIZE

history = model.fit(train.repeat(),
                    epochs=10,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation.repeat(),
                    validation_steps=validation_steps)

"""## Training (QAT)"""

q_aware_model = tfmo.quantization.keras.quantize_model(model)

q_aware_model.summary()

q_aware_model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                      loss = 'sparse_categorical_crossentropy',
                      metrics = ["accuracy"])

q_aware_history = q_aware_model.fit(train.repeat(),
                                    initial_epoch=10,
                                    epochs=20,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_data=validation.repeat(),
                                    validation_steps=validation_steps)

acc = history.history['accuracy'] + q_aware_history.history['accuracy']
val_acc = history.history['val_accuracy'] + q_aware_history.history['val_accuracy']

loss = history.history['loss'] + q_aware_history.history['loss']
val_loss = history.history['val_loss'] + q_aware_history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(10, ls='-.', color='magenta')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axvline(10, ls='-.', color='magenta')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

"""## Evaluate model"""

_, baseline_model_accuracy = model.evaluate(test)
_, q_aware_model_accuracy = q_aware_model.evaluate(test)

print('Baseline model test accuracy:', baseline_model_accuracy)
print('QAT model test accuracy   :', q_aware_model_accuracy)

"""# Convert Keras model to TF-Lite model"""

models_dir = pathlib.Path(os.path.join('.', 'models'))
models_dir.mkdir(exist_ok=True, parents=True)

"""## TF-Lite model"""

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.experimental_new_converter = True
tflite_model = converter.convert()
with open(os.path.join(models_dir, 'mobilenet_v2.tflite'), 'wb') as f:
    f.write(tflite_model)

"""## Weight quantization model (Post training quantization)"""

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_weight_quant_model = converter.convert()
with open(os.path.join(models_dir, 'mobilenet_v2_weight_quant.tflite'), 'wb') as f:
    f.write(tflite_weight_quant_model)

"""## Float16 quantization model (Post training quantization)"""

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16_quant_model = converter.convert()
with open(os.path.join(models_dir, 'mobilenet_v2_fp16_quant.tflite'), 'wb') as f:
    f.write(tflite_fp16_quant_model)

"""## Integer quantization model (Post training quantization)"""

def representative_data_gen():
  for batch in test.take(255):
    yield [batch[0]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
tflite_full_integer_quant_model = converter.convert()

with open(os.path.join(models_dir, 'mobilenet_v2_integer_quant.tflite'), 'wb') as f:
    f.write(tflite_full_integer_quant_model)

"""## Integer quantization model (Quantization aware training)"""

#with tfmo.quantization.keras.quantize_scope():
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)

converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_q_aware_integer_quant_model = converter.convert()
with open(os.path.join(models_dir, 'mobilenet_v2_q_aware_integer_quant.tflite'), 'wb') as f:
    f.write(tflite_q_aware_integer_quant_model)

"""# Save models (keras models)"""

# Save keras model
model.save(os.path.join(models_dir, 'mobilenet_v2.h5'))
q_aware_model.save(os.path.join(models_dir, 'mobilenet_v2_quant.h5'))

"""# Inference TF-Lite model"""

def inference_tflite(mode_path, num_test):
  interpreter = tf.lite.Interpreter(model_path=mode_path)

  interpreter.allocate_tensors()
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  print('input_details:  ', interpreter.get_input_details())
  print('output_details: ', interpreter.get_output_details())

  total_seen = 0
  num_correct = 0
  inference_time = []

  for batch in test.take(int(num_test)):
    image = batch[0].numpy()

    start_ms = time.time()

    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])

    elapsed_ms = time.time() - start_ms
    inference_time.append(elapsed_ms * 1000.0)

    if batch[1].numpy() == np.argmax(predictions):
      num_correct += 1
    total_seen += 1

    if total_seen % 50 == 0:
        print("Accuracy after %i images: %f" %
              (total_seen, float(num_correct) / float(total_seen)))

  print('Num images: {0:}, Accuracy: {1:.4f}, Latency: {2:.2f} ms'.format(num_test,
                                                                         float(num_correct / total_seen),
                                                                         np.array(inference_time).mean()))

"""Evaluate TF-Lite model"""

model_path = os.path.join(models_dir, 'mobilenet_v2.tflite')
inference_tflite(model_path, int(num_test))

"""Evaluate weight quantization model (Post training quantization)"""

model_path = os.path.join(models_dir, 'mobilenet_v2_weight_quant.tflite')
inference_tflite(model_path, int(num_test))

"""Evaluate Float16 quantization model (Post training quantization)"""

model_path = os.path.join(models_dir, 'mobilenet_v2_fp16_quant.tflite')
inference_tflite(model_path, int(num_test))

"""Evaluate Integer quantization model (Post training quantization)"""

model_path = os.path.join(models_dir, 'mobilenet_v2_integer_quant.tflite')
inference_tflite(model_path, int(num_test))

"""Evaluate Integer quantization model (Quantization aware training)"""

model_path = os.path.join(models_dir, 'mobilenet_v2_q_aware_integer_quant.tflite')
inference_tflite(model_path, int(num_test))

"""# Compile Edge TPU Model"""



#!echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
#!sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6A030B21BA07F4FB
#!sudo apt update -qq
#!sudo apt install edgetpu-compiler
#
#!edgetpu_compiler -v
#
#"""Compile integer quantization (Post training quantization)"""
#
#!edgetpu_compiler -s --out_dir /content/models /content/models/mobilenet_v2_integer_quant.tflite
#
#"""Compile integer quantization (Quantization aware training)"""
#
#!edgetpu_compiler -s --out_dir /content/models /content/models/mobilenet_v2_q_aware_integer_quant.tflite
