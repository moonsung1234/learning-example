
from tensorflow import keras
import matplotlib.pyplot as plt

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
input1 = train_input[0].reshape(-1, 28, 28, 1) / 255

model = keras.models.load_model("best-cnn1.h5")
conv_acti = keras.Model(model.input, model.layers[0].output)
conv2_acti = keras.Model(model.input, model.layers[2].output)

conv_acti_predicted_value = conv_acti.predict(input1)
conv2_acti_predicted_value = conv2_acti.predict(input1)

fig, axs = plt.subplots(4, 8, figsize=(15, 8))

for i in range(4) :
    for j in range(8) :
        axs[i, j].imshow(conv_acti_predicted_value[0, :, :, i * 8 + j])
        axs[i, j].axis("off")

plt.show()

fig, axs = plt.subplots(8, 8, figsize=(12, 12))

for i in range(8) :
    for j in range(8) :
        axs[i, j].imshow(conv2_acti_predicted_value[0, :, :, i * 8 + j])
        axs[i, j].axis("off")

plt.show()
