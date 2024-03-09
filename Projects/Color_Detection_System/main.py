import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
from keras import layers

dataset = pd.read_csv('Color_Detection_System/final_data.csv')
dataset.label.unique()
dataset = pd.get_dummies(dataset, columns=['label'])
train_dataset = dataset.sample(frac=0.8, random_state=8)
test_dataset = dataset.drop(train_dataset.index)
train_labels = pd.DataFrame([train_dataset.pop(x) for x in ['label_Black', 'label_Blue', 'label_Brown',
       'label_Green', 'label_Grey', 'label_Orange', 'label_Pink',
       'label_Purple', 'label_Red', 'label_White', 'label_Yellow']]).T
test_labels = pd.DataFrame([test_dataset.pop(x) for x in ['label_Black', 'label_Blue', 'label_Brown',
       'label_Green', 'label_Grey', 'label_Orange', 'label_Pink',
       'label_Purple', 'label_Red', 'label_White', 'label_Yellow']]).T

from keras import regularizers
model = keras.Sequential([
    layers.Dense(3, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(24, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    layers.Dense(24, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    layers.Dense(11)
])
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy'])
history = model.fit(x=train_dataset, y=train_labels, 
                    validation_split=0.2, 
                    epochs=5001, 
                    batch_size=2048, 
                    verbose=0,
                    callbacks=[tfdocs.modeling.EpochDots()], 
                    shuffle=True)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()