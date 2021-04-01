
import numpy as np
import matplotlib.pyplot as plt

fruits = np.load("fruits.npy")

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
# axs[0].imshow(fruits[0], cmap="gray_r")
# axs[1].imshow(fruits[100], cmap="gray_r")
# axs[2].imshow(fruits[200], cmap="gray_r")
# plt.show()

apples = fruits[0:100].reshape(-1, 100 * 100)
pineapples = fruits[100:200].reshape(-1, 100 * 100)
bananas = fruits[200:300].reshape(-1, 100 * 100)

# plt.hist(np.mean(apples, axis=1))
# plt.hist(np.mean(pineapples, axis=1))
# plt.hist(np.mean(bananas, axis=1))
# plt.show()

# axs[0].bar(range(100 * 100), np.mean(apples, axis=0))
# axs[1].bar(range(100 * 100), np.mean(pineapples, axis=0))
# axs[2].bar(range(100 * 100), np.mean(bananas, axis=0))
# plt.show()

apple_mean = np.mean(apples, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapples, axis=0).reshape(100, 100)
banana_mean = np.mean(bananas, axis=0).reshape(100, 100)

# axs[0].imshow(apple_mean, cmap="gray_r")
# axs[1].imshow(pineapple_mean, cmap="gray_r")
# axs[2].imshow(banana_mean, cmap="gray_r")
# plt.show()

abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))
apple_index = np.argsort(abs_mean)[:100]

fig, axs = plt.subplots(10, 10, figsize=(10, 10))

for i in range(10) :
    for j in range(10) :
        axs[i, j].imshow(fruits[apple_index[i * 10 + j]], cmap="gray_r")
        axs[i, j].axis("off")

plt.show()
