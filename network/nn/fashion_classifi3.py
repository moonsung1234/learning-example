
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=2021)

def getModel(layer=None) :
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation="relu"))

    if layer :
        model.add(layer)

    model.add(keras.layers.Dense(10, activation="softmax"))

    return model

model = getModel(keras.layers.Dropout(0.3))
model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
checkpoint = keras.callbacks.ModelCheckpoint("best_model.h5")
stoppoint = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=5, verbose=1, validation_data=(val_scaled, val_target), callbacks=[checkpoint, stoppoint])

# plt.plot(history.history["loss"])
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.show()

# plt.plot(history.history["accuracy"])
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epochs")
plt.ylabel("loss(all)")
plt.legend(["train", "val"])
plt.show()