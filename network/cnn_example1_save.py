
from tensorflow import keras
import matplotlib.pyplot as plt

model = keras.models.load_model("best-cnn1.h5")
conv_weights = model.layers[0].weights[0].numpy()

plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel("weight")
plt.ylabel("count")
plt.show()

fig, axs = plt.subplots(2, 16, figsize=(16, 2))

for i in range(2) :
    for j in range(16) :
        axs[i, j].imshow(conv_weights[:, :, 0, i * 16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis("off")

plt.show()