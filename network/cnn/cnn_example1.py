
from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=2021)

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, activation="relu", padding="same", input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(2)) # final shape is (14, 14, 32)
model.add(keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"))
model.add(keras.layers.MaxPooling2D(2)) # final shape is (7, 7, 64)
model.add(keras.layers.Flatten(input_shape=(7, 7, 64)))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")

checkpoint = keras.callbacks.ModelCheckpoint("best-cnn1.h5")
stoppoint = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=10, validation_data=(val_scaled, val_target), callbacks=(checkpoint, stoppoint))

# keras.utils.plot_model(model)



