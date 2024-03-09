import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras.layers as kl

from keras.preprocessing import image_dataset_from_directory
from keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

batch_size = 32
img_size = (160, 160)
directory = "dataset"

train_dataset = image_dataset_from_directory(directory,
                                             shuffle = True,
                                             batch_size=batch_size,
                                             image_size=img_size,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)

validation_dataset = image_dataset_from_directory(directory,
                                                  shuffle=True,
                                                  batch_size=batch_size,
                                                  image_size=img_size,
                                                  validation_split=0.2,
                                                  subset='validation',
                                                  seed=42)

autotune = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffersize = autotune)

def data_augmenter():
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip('horizontal'))
    data_augmentation.add(RandomRotation(0.2))

    return data_augmentation

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

img_shape = img_size + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                               include_top=True,
                                               weights='imagenet')

def alpaca_model(image_shape = img_size, data_augmentation = data_augmenter()):
    input_shape = image_shape + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape = input_shape,
                                                   include_top = False,
                                                   weights = 'imagenet')
    
    base_model.trainable = False

    inputs = tf.keras.Input(shape = input_shape)

    x = data_augmentation(inputs)
    x = preprocess_input(x)

    x = base_model(x, training = False)
    x = kl.GlobalAveragePooling2D()(x)
    x = kl.Dropout(0.2)(x)

    prediction_layer = kl.Dense(1)
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    return model

model2 = alpaca_model(img_size, data_augmenter)

base_learning_rate = 0.001
model2.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
               loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
               metrics = ['accuracy'])

history = model2.fit(train_dataset, validation_data=validation_dataset, epochs = 5)

acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

base_model = model2.layers[4]
base_model.trainable = True

fine_tune_at = 120

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = None
loss_function=tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate*0.1)
metrics=['accuracy']

model2.compile(loss=loss_function,
              optimizer = optimizer,
              metrics=metrics)

history_fine = model2.fit(train_dataset,
                          epochs = 10,
                          initial_epoch=history.epoch[-1],
                          validation_data=validation_dataset)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
plt.plot([4, 4],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([4, 4],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()