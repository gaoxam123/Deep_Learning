import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

tf.random.set_seed(1234)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (28 * 28, )),
    tf.keras.layers.Dense(units = 100, activation='relu'),
    tf.keras.layers.Dense(units = 50, activation='relu'),
    tf.keras.layers.Dense(units = 10, activation='linear')
])


x_train = x_train.reshape(x_train.shape[0], 28 * 28, 1)
x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.reshape(x_test.shape[0], 28 * 28, 1)
x_test = x_test.astype('float32')
x_test /= 255

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy'])
model.fit(x_train[0:1000], y_train[0:1000], epochs = 40)
output = model.predict(x_test[10:20])
out = np.zeros(10)
for i in range(output.shape[0]):
    out[i] = np.argmax(output[i])
print(out, end = '\n')
print(y_test[10:20])