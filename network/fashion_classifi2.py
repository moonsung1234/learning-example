
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=2021)

flatten = keras.layers.Flatten(input_shape=(28, 28))
dense1 = keras.layers.Dense(100, activation="relu")
dense2 = keras.layers.Dense(10, activation="softmax")

model = keras.Sequential([flatten, dense1, dense2])
model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)

