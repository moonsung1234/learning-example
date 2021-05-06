
from tensorflow import keras
import numpy as np

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255
train_scaled = train_scaled.reshape(-1, 28 * 28)
test_scaled = test_input / 255
test_scaled = test_scaled.reshape(-1, 28 * 28)

# print(np.unique(train_target, return_counts=True))

dense = keras.layers.Dense(10, activation="softmax", input_shape=(784,))
model = keras.Sequential(dense)
model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(test_scaled, test_target)